"""Turn HTML / Markdown / MDX into clean text blocks with source mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

BLOCK_TAGS = {
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "td", "th", "figcaption", "dd", "dt", "summary",
}
DROP_TAGS = {
    "script", "style", "noscript", "svg", "iframe", "nav", "footer",
    "header", "aside", "form", "button", "select", "template",
}

_WS = re.compile(r"\s+")


@dataclass
class Block:
    text: str
    kind: str
    index: int
    raw: str | None = None
    line: int | None = None

    def as_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "index": self.index, "line": self.line}


@dataclass
class Doc:
    title: str
    description: str
    blocks: list[Block]
    links: list[str]
    h1: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def words(self) -> int:
        return len(self.text.split())

    def headings(self) -> list[str]:
        return [b.text for b in self.blocks if b.kind.startswith("h")]


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def parse_html(html: str) -> Doc:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = _clean(soup.title.string)

    description = ""
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if meta and meta.get("content"):
        description = _clean(meta["content"])
    if not description:
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            description = _clean(og["content"])

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
            links.append(href)

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()

    blocks: list[Block] = []
    seen: set[str] = set()
    for el in soup.find_all(BLOCK_TAGS):
        # skip blocks nested inside another block we already captured
        if any(p.name in BLOCK_TAGS for p in el.parents):
            continue
        text = _clean(el.get_text(" ", strip=True))
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        if len(text) < 12:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        blocks.append(Block(text=text, kind=el.name, index=len(blocks)))

    h1 = next((b.text for b in blocks if b.kind == "h1"), "")
    return Doc(title=title, description=description, blocks=blocks, links=links, h1=h1)


def parse_markdown(md: str) -> Doc:
    lines = md.splitlines()

    # strip YAML frontmatter
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), 80)):
            if lines[i].strip() == "---":
                start = i + 1
                break

    blocks: list[Block] = []
    links: list[str] = []
    in_fence = False
    para: list[str] = []
    para_line = 0

    def flush() -> None:
        nonlocal para
        if not para:
            return
        text = _clean(" ".join(para))
        para = []
        if text and re.search(r"[A-Za-z0-9]", text) and len(text) >= 12:
            blocks.append(Block(text=text, kind="p", index=len(blocks), line=para_line))

    for i in range(start, len(lines)):
        raw = lines[i]
        s = raw.strip()

        if s.startswith("```") or s.startswith("~~~"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not s:
            flush()
            continue
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
            flush()
            continue
        if s.startswith("import ") or s.startswith("export ") or s.startswith("<"):
            flush()
            continue

        links.extend(re.findall(r"\]\((https?://[^)\s]+)\)", s))
        links.extend(re.findall(r"href=[\"']([^\"']+)[\"']", s))

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            flush()
            blocks.append(
                Block(text=_clean(m.group(2)), kind=f"h{len(m.group(1))}",
                      index=len(blocks), line=i + 1)
            )
            continue
        if re.match(r"^([-*+]|\d+[.)])\s+", s):
            flush()
            text = re.sub(r"^([-*+]|\d+[.)])\s+", "", s)
            text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
            text = re.sub(r"[*_`]{1,3}(.+?)[*_`]{1,3}", r"\1", text)
            text = _clean(text)
            if text and len(text) >= 12:
                blocks.append(Block(text=text, kind="li", index=len(blocks), line=i + 1))
            continue

        if not para:
            para_line = i + 1
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[*_`]{1,3}(.+?)[*_`]{1,3}", r"\1", text)
        para.append(text)
    flush()

    title = next((b.text for b in blocks if b.kind == "h1"), "")
    h1 = title
    return Doc(title=title, description="", blocks=blocks, links=links, h1=h1)


def parse_file(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    if suffix in {".md", ".mdx", ".markdown"}:
        return parse_markdown(raw)
    return parse_html(raw)


def chunk_blocks(blocks: list[Block], target_words: int = 240) -> list[list[Block]]:
    """Group consecutive blocks so the model sees ~one section of context."""
    chunks: list[list[Block]] = []
    cur: list[Block] = []
    words = 0
    for b in blocks:
        w = len(b.text.split())
        if cur and words + w > target_words:
            chunks.append(cur)
            cur, words = [], 0
        cur.append(b)
        words += w
    if cur:
        chunks.append(cur)
    return chunks
