"""Run LinkBERT over a page and map spans back to source blocks."""

from __future__ import annotations

from .extract import Block, chunk_blocks
from .model import LinkPredictor, Span
from .site import Page


def analyze_page(
    predictor: LinkPredictor,
    page: Page,
    high: float = 0.50,
    low: float = 0.22,
    min_words: int = 1,
    max_words: int = 8,
    include_headings: bool = False,
) -> list[Span]:
    blocks = page.blocks
    usable = [
        b for b in blocks
        if include_headings or not b.kind.startswith("h")
    ]
    if not usable:
        return []

    spans: list[Span] = []
    for chunk in chunk_blocks(usable, target_words=240):
        text = "\n\n".join(b.text for b in chunk)
        bounds: list[tuple[int, int, Block]] = []
        pos = 0
        for b in chunk:
            bounds.append((pos, pos + len(b.text), b))
            pos += len(b.text) + 2

        for sp in predictor.predict(text, high=high, low=low,
                                    min_words=min_words, max_words=max_words):
            for start, end, block in bounds:
                if start <= sp.start and sp.end <= end:
                    spans.append(
                        Span(
                            text=sp.text,
                            start=sp.start - start,
                            end=sp.end - start,
                            score=sp.score,
                            words=sp.words,
                            sentence=sp.sentence,
                            paragraph=block.index,
                        )
                    )
                    break

    spans.sort(key=lambda s: (s.paragraph or 0, s.start))
    return spans


def span_block(page: Page, span: Span) -> Block | None:
    if span.paragraph is None:
        return None
    return next((b for b in page.blocks if b.index == span.paragraph), None)
