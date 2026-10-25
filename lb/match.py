"""Match predicted anchor spans to the page they should point at."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import numpy as np

from .model import Span
from .site import Page, SiteIndex

STOP = set(
    """a an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being it its you your we our they their he she his her i me my
    not no do does did can could should would will just about into over under more most very
    how what when where why which who whom whose also may might must shall""".split()
)

WEIGHT_SEMANTIC = 0.60
WEIGHT_HEADING = 0.27
WEIGHT_SLUG = 0.13


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def _slug(key: str) -> str:
    return re.sub(r"[-_/]+", " ", key).strip()


def resolve(href: str, page: Page, idx: SiteIndex) -> Page | None:
    """Map an href to an indexed page, tolerating relative paths and trailing slashes."""
    import posixpath

    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
        return None

    if href.startswith(("http://", "https://")):
        parsed = urlparse(href)
        path = parsed.path or "/"
    else:
        path = href.split("#")[0].split("?")[0]
        if not path:
            return None
        if not path.startswith("/"):
            base = page.key if page.key.endswith("/") else page.key.rsplit("/", 1)[0] + "/"
            path = posixpath.normpath(base + path)
            if href.endswith("/") and not path.endswith("/"):
                path += "/"

    path = re.sub(r"/index\.html?$", "/", path)
    path = re.sub(r"\.html?$", "", path)
    if not path.startswith("/"):
        path = "/" + path

    candidates = [path]
    candidates.append(path.rstrip("/") or "/")
    candidates.append(path.rstrip("/") + "/")
    for c in dict.fromkeys(candidates):
        if c in idx.by_key:
            return idx.by_key[c]
    for k in idx.by_key:
        if k.rstrip("/") == path.rstrip("/"):
            return idx.by_key[k]
    return None


@dataclass
class Suggestion:
    source_key: str
    source_title: str
    source_file: str
    anchor: str
    sentence: str
    target_key: str
    target_title: str
    target_url: str
    score: float
    components: dict = field(default_factory=dict)
    start: int = 0
    end: int = 0
    block: int | None = None
    line: int | None = None

    def as_dict(self) -> dict:
        return {
            "source": self.source_key,
            "source_file": self.source_file,
            "anchor": self.anchor,
            "sentence": self.sentence,
            "target": self.target_key,
            "target_title": self.target_title,
            "target_url": self.target_url,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "block": self.block,
            "line": self.line,
        }


class Matcher:
    def __init__(self, idx: SiteIndex, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.idx = idx
        if idx.vectors is None:
            idx.embed(model_name)
        self.model = SentenceTransformer(model_name)
        self._head_tokens = [
            _tokens(" ".join([p.title, p.h1] + p.headings)) for p in idx.pages
        ]
        self._slug_tokens = [_tokens(_slug(p.key)) for p in idx.pages]

    def _rank(self, anchor: str, sentence: str, source: Page, exclude: set[str]) -> list[tuple[Page, float, dict]]:
        q = self.model.encode(
            [anchor + ". " + sentence[:220]], normalize_embeddings=True
        )[0].astype("float32")
        sims = self.idx.vectors @ q
        at = _tokens(anchor)
        ranked: list[tuple[Page, float, dict]] = []
        for i, page in enumerate(self.idx.pages):
            if page.key == source.key or page.key in exclude:
                continue
            if page.words < 40:
                continue
            cos = float(sims[i])
            head = len(at & self._head_tokens[i]) / len(at) if at else 0.0
            slug = len(at & self._slug_tokens[i]) / len(at) if at else 0.0
            score = WEIGHT_SEMANTIC * cos + WEIGHT_HEADING * head + WEIGHT_SLUG * slug
            ranked.append((page, score, {"semantic": cos, "heading": head, "slug": slug}))
        ranked.sort(key=lambda t: -t[1])
        return ranked

    def for_spans(
        self,
        page: Page,
        spans: list[Span],
        min_score: float = 0.34,
        per_target: int = 2,
        skip_linked: bool = True,
    ) -> list[Suggestion]:
        linked: set[str] = set()
        if skip_linked:
            for href in page.links:
                t = resolve(href, page, self.idx)
                if t:
                    linked.add(t.key)

        out: list[Suggestion] = []
        used_targets: dict[str, int] = {}
        for sp in spans:
            ranked = self._rank(sp.text, sp.sentence, page, linked)
            if not ranked:
                continue
            target, score, comps = ranked[0]
            if score < min_score:
                continue
            if used_targets.get(target.key, 0) >= per_target:
                continue
            used_targets[target.key] = used_targets.get(target.key, 0) + 1
            out.append(
                Suggestion(
                    source_key=page.key,
                    source_title=page.title,
                    source_file=page.source,
                    anchor=sp.text,
                    sentence=sp.sentence,
                    target_key=target.key,
                    target_title=target.title,
                    target_url=target.source,
                    score=score,
                    components=comps,
                    start=sp.start,
                    end=sp.end,
                    block=sp.paragraph,
                )
            )
        out.sort(key=lambda s: -s.score)
        return out

    # ------------------------------------------------------------- reverse
    def score_against(self, anchor: str, sentence: str, target_vec, head_tokens: set[str],
                      slug_tokens: set[str]) -> tuple[float, dict]:
        """Score one anchor against one target page profile."""
        q = self.model.encode(
            [anchor + ". " + sentence[:220]], normalize_embeddings=True
        )[0].astype("float32")
        cos = float(target_vec @ q)
        at = _tokens(anchor)
        head = len(at & head_tokens) / len(at) if at else 0.0
        slug = len(at & slug_tokens) / len(at) if at else 0.0
        return (
            WEIGHT_SEMANTIC * cos + WEIGHT_HEADING * head + WEIGHT_SLUG * slug,
            {"semantic": cos, "heading": head, "slug": slug},
        )

    def gaps(self, min_score: float = 0.40, sources_per_page: int = 6,
             max_pages: int | None = None) -> list[dict]:
        """Pages with the fewest inbound internal links, plus where to link from."""
        inbound: dict[str, set[str]] = {p.key: set() for p in self.idx.pages}
        for p in self.idx.pages:
            for href in p.links:
                t = resolve(href, p, self.idx)
                if t and t.key != p.key:
                    inbound[t.key].add(p.key)

        ranked = sorted(self.idx.pages, key=lambda p: (len(inbound[p.key]), -p.words))
        if max_pages:
            ranked = ranked[:max_pages]

        pos = {p.key: i for i, p in enumerate(self.idx.pages)}
        results = []
        for page in ranked:
            i = pos[page.key]
            sims = self.idx.vectors @ self.idx.vectors[i]
            order = np.argsort(-sims)
            sources = [
                self.idx.pages[j] for j in order if self.idx.pages[j].key != page.key
            ][:sources_per_page]
            results.append({
                "page": page.key,
                "title": page.title,
                "inbound": len(inbound[page.key]),
                "words": page.words,
                "sources": [
                    {"key": s.key, "title": s.title, "semantic": float(sims[pos[s.key]])}
                    for s in sources
                ],
            })
        return results
