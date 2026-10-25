"""Discover, fetch and index a site (local folder or live URL)."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests

from .extract import Block, Doc, parse_html, parse_file

UA = "linkbert-seo/1.0 (+internal linking analysis)"
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}

LOCAL_HTML = {".html", ".htm"}
LOCAL_MD = {".md", ".mdx", ".markdown"}
SKIP_DIRS = {
    "node_modules", ".git", ".next", ".nuxt", ".cache", "vendor",
    "assets", "images", "img", "fonts", "static", "media",
}
BUILD_DIRS = ["dist", "out", "build", "_site", "public", ".output", "site"]


@dataclass
class Page:
    key: str                 # canonical site path, e.g. /blog/post/
    source: str              # absolute file path or URL
    title: str = ""
    description: str = ""
    h1: str = ""
    headings: list[str] = field(default_factory=list)
    text: str = ""
    words: int = 0
    links: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    is_draft: bool = False

    def embed_text(self) -> str:
        parts = [self.title, self.h1, " | ".join(self.headings[:12])]
        if self.description:
            parts.append(self.description)
        parts.append(self.text[:900])
        return ". ".join(p for p in parts if p)

    def as_dict(self) -> dict:
        return {
            "key": self.key, "source": self.source, "title": self.title,
            "description": self.description, "h1": self.h1,
            "headings": self.headings, "text": self.text, "words": self.words,
            "links": self.links,
            "blocks": [b.as_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        blocks = [Block(**b) for b in d.pop("blocks", [])]
        return cls(blocks=blocks, **d)


# --------------------------------------------------------------------- local
def _local_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    rel = re.sub(r"index\.(html?|mdx?|markdown)$", "", rel)
    rel = re.sub(r"\.(html?|mdx?|markdown)$", "", rel)
    if not rel.startswith("/"):
        rel = "/" + rel
    if not rel.endswith("/") and "." not in rel.rsplit("/", 1)[-1]:
        rel += "/"
    return rel or "/"


def _pick_root(target: Path) -> Path:
    html_at_root = list(target.glob("*.html")) + list(target.glob("*.md"))
    if html_at_root:
        return target
    for name in BUILD_DIRS:
        cand = target / name
        if cand.is_dir() and (list(cand.rglob("*.html")) or list(cand.rglob("*.md"))):
            return cand
    return target


def load_local(target: Path, limit: int | None = None) -> list[Page]:
    root = _pick_root(target)
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in LOCAL_HTML | LOCAL_MD:
            continue
        if any(part.lower() in SKIP_DIRS for part in p.parts):
            continue
        if p.name.lower() in {"404.html", "500.html", "_app.html"}:
            continue
        files.append(p)
    if limit:
        files = files[:limit]

    pages: list[Page] = []
    for f in files:
        try:
            doc: Doc = parse_file(f)
        except Exception:
            continue
        if not doc.blocks:
            continue
        pages.append(
            Page(
                key=_local_key(f, root),
                source=str(f),
                title=doc.title or doc.h1,
                description=doc.description,
                h1=doc.h1,
                headings=doc.headings(),
                text=doc.text,
                words=doc.words,
                links=doc.links,
                blocks=doc.blocks,
            )
        )
    return pages


# ----------------------------------------------------------------------- web
def _get(url: str, timeout: int = 25) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                return r
            if r.status_code in {403, 404, 410}:
                return None
        except requests.RequestException:
            pass
        time.sleep(0.6 * (attempt + 1))
    return None


def _sitemap_urls(base: str, max_pages: int) -> list[str]:
    origin = "{0.scheme}://{0.netloc}".format(urlparse(base))
    seeds: list[str] = []
    robots = _get(origin + "/robots.txt")
    if robots is not None:
        seeds += re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots.text)
    seeds += [
        origin + "/sitemap.xml",
        origin + "/sitemap_index.xml",
        origin + "/wp-sitemap.xml",
        origin + "/sitemap-index.xml",
    ]

    found: list[str] = []
    seen: set[str] = set()

    def walk(url: str, depth: int = 0) -> None:
        if url in seen or depth > 2 or len(found) >= max_pages:
            return
        seen.add(url)
        r = _get(url)
        if r is None:
            return
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
        if re.search(r"<sitemapindex", r.text, re.I):
            for loc in locs:
                walk(loc.strip(), depth + 1)
        else:
            for loc in locs:
                loc = loc.strip()
                if loc.lower().endswith((".xml", ".gz")):
                    walk(loc, depth + 1)
                elif loc not in found:
                    found.append(loc)
                    if len(found) >= max_pages:
                        return

    for s in dict.fromkeys(seeds):
        if found:
            break
        walk(s)
    return found[:max_pages]


def _crawl_links(base: str, max_pages: int, max_depth: int) -> list[str]:
    origin = "{0.scheme}://{0.netloc}".format(urlparse(base))
    seen = {base}
    order = [base]
    depth = {base: 0}
    i = 0
    while i < len(order) and len(order) < max_pages:
        url = order[i]
        i += 1
        if depth[url] >= max_depth:
            continue
        r = _get(url)
        if r is None:
            continue
        for href in re.findall(r'href=["\']([^"\']+)["\']', r.text):
            nxt = urldefrag(urljoin(url, href))[0]
            if not nxt.startswith(origin):
                continue
            if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|pdf|zip|css|js|xml|json)$", nxt, re.I):
                continue
            if nxt not in seen:
                seen.add(nxt)
                depth[nxt] = depth[url] + 1
                order.append(nxt)
                if len(order) >= max_pages:
                    break
    return order[:max_pages]


def discover(base: str, max_pages: int = 200, max_depth: int = 3) -> list[str]:
    urls = _sitemap_urls(base, max_pages)
    if len(urls) < 2:
        urls = _crawl_links(base, max_pages, max_depth)
    return urls


def load_web(base: str, max_pages: int = 200, max_depth: int = 3) -> list[Page]:
    urls = discover(base, max_pages, max_depth)
    pages: list[Page] = []

    def one(url: str) -> Page | None:
        r = _get(url)
        if r is None or "html" not in r.headers.get("content-type", "html"):
            return None
        try:
            doc = parse_html(r.text)
        except Exception:
            return None
        if not doc.blocks:
            return None
        path = urlparse(url).path or "/"
        return Page(
            key=path, source=url, title=doc.title or doc.h1,
            description=doc.description, h1=doc.h1, headings=doc.headings(),
            text=doc.text, words=doc.words, links=doc.links, blocks=doc.blocks,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(one, u): u for u in urls}
        for fut in as_completed(futures):
            page = fut.result()
            if page:
                pages.append(page)

    pages.sort(key=lambda p: p.key)
    return pages


# --------------------------------------------------------------------- store
class SiteIndex:
    def __init__(self, pages: list[Page], root: str = "", kind: str = "local") -> None:
        self.pages = pages
        self.root = root
        self.kind = kind
        self.by_key = {p.key: p for p in pages}
        self.vectors = None  # np.ndarray set by embed()

    def __len__(self) -> int:
        return len(self.pages)

    def embed(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = [p.embed_text() for p in self.pages]
        self.vectors = model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).astype("float32")

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "site.json").write_text(
            json.dumps(
                {
                    "root": self.root,
                    "kind": self.kind,
                    "pages": [p.as_dict() for p in self.pages],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if self.vectors is not None:
            import numpy as np

            np.save(out / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, out: Path) -> "SiteIndex":
        data = json.loads((out / "site.json").read_text(encoding="utf-8"))
        idx = cls([Page.from_dict(p) for p in data["pages"]], data.get("root", ""), data.get("kind", "local"))
        vec = out / "vectors.npy"
        if vec.exists():
            import numpy as np

            idx.vectors = np.load(vec)
        return idx


def build(target: str, out: Path, max_pages: int = 200, max_depth: int = 3,
          embed: bool = True) -> SiteIndex:
    if target.startswith(("http://", "https://")):
        pages = load_web(target, max_pages=max_pages, max_depth=max_depth)
        idx = SiteIndex(pages, root=target, kind="web")
    else:
        root = Path(target).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        pages = load_local(root, limit=max_pages)
        idx = SiteIndex(pages, root=str(root), kind="local")
    if not pages:
        raise RuntimeError(f"No pages found at {target}")
    if embed:
        idx.embed()
    idx.save(out)
    return idx
