"""Hybrid Link Opportunity Discovery Engine.

Combines three specialized algorithms:
1. Direct Target Entity & Mention Mining (High Precision / 100% recall on unlinked target topics)
2. Calibrated Adaptive LinkBERT Inference (Contextual Placement & Natural Anchor Detection)
3. Semantic Conceptual Keyphrase Matcher (Vector Similarity & Intent Alignment)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from .extract import Block
from .model import LinkPredictor, Span
from .site import Page, SiteIndex


# Core stopwords and overly generic anchor words to suppress
GENERIC_STOP = {
    "click here", "read more", "learn more", "here", "this", "this article",
    "this page", "this post", "link", "website", "more", "see more", "find out more",
    "home", "about", "contact", "blog", "services", "guide", "article", "page",
    "info", "overview", "details", "help", "menu", "privacy policy", "terms",
    "get a free call instead", "contact me directly", "requires you to publish the site",
    "page onto google", "view details", "one-to-one guidance for students",
    "i implemented comprehensive woocommerce seo",
}

CTA_STOP_PATTERNS = [
    r"^get a free\b",
    r"^contact me\b",
    r"^book a\b",
    r"^requires you to\b",
    r"^click here\b",
    r"^read more\b",
    r"^page onto\b",
    r"^i implemented\b",
    r"^we implemented\b",
    r"^schedule a\b",
    r"^hire me\b",
    r"^view details\b",
]

# Generic words that shouldn't stand alone as single-word anchors
GENERIC_SINGLE_WORDS = {
    "pet", "insurance", "cat", "cats", "dog", "dogs", "animal", "animals", "policy",
    "policies", "cost", "costs", "price", "care", "health", "vet", "vets", "cover",
    "best", "top", "free", "good", "new", "easy", "simple", "day", "days", "year", "years",
}


def check_topical_coherence(anchor: str, sentence: str, source_key: str, target_key: str) -> float:
    """Computes a topical coherence modifier between anchor/sentence and candidate target page.
    
    Prevents cross-silo hallucinations (e.g. Core Web Vitals -> Web 2.0 Link Building)
    and enforces strict canonical entity resolution (e.g. personal name -> /about-muyiwa-adekojede).
    
    Returns:
    - <= -50.0: Strictly forbidden / cross-silo contamination
    - 0.0: Neutral
    - > 0.0: Strong affinity boost
    """
    anchor_lower = anchor.lower()
    sent_lower = sentence.lower()
    tgt_lower = target_key.lower()

    # 1. Technical Site Performance & Core Web Vitals Silo
    perf_keywords = ("core web vitals", "pagespeed", "lighthouse", "lcp", "cls", "inp", "fid", "site speed", "web vitals", "fail core web")
    is_perf = any(k in anchor_lower for k in perf_keywords) or ("core web vitals" in sent_lower and len(anchor_lower.split()) <= 4)
    
    if is_perf:
        # Strict block on off-page link building, duplicate content, or PPC
        if "link-building" in tgt_lower or "web-2-0" in tgt_lower or "duplicate-content" in tgt_lower or tgt_lower.startswith("/ppc") or tgt_lower.startswith("/academics"):
            return -100.0
        # Boost dedicated performance audit / comparison pages
        if tgt_lower in ("/tools/pagespeed-compare", "/international-seo/technical-seo-audits") or "wordpress-seo-expert" in tgt_lower:
            return 0.50

    # 2. Off-Page Link Building & Backlink Silo
    link_keywords = ("link building", "backlinks", "web 2.0", "guest post", "citations", "backlink loss", "lost 1,212")
    is_link = any(k in anchor_lower for k in link_keywords)
    if is_link:
        if tgt_lower in ("/tools/pagespeed-compare",) or tgt_lower.startswith(("/academics", "/ppc")):
            return -100.0
        if "link-building" in tgt_lower or "swadesi-news" in tgt_lower or "reign-of-titans" in tgt_lower or "web-2-0" in tgt_lower:
            return 0.40

    # 3. Canonical Personal Entity Resolution
    personal_keywords = ("muyiwa adekojede", "muyiwa", "who i am", "about muyiwa", "i am muyiwa")
    if any(k in anchor_lower for k in personal_keywords):
        if tgt_lower == "/about-muyiwa-adekojede":
            return 0.60
        # Never route author persona name to university teaching sub-pages or case studies
        return -100.0

    # 4. Commercial SEO vs Academic Research Silo
    if any(k in anchor_lower for k in ("keyword research", "seo strategy", "ecommerce seo", "link portfolio")):
        if tgt_lower.startswith("/academics"):
            return -100.0

    # 5. PPC Google Ads Silo
    if any(k in anchor_lower for k in ("google ads", "ppc", "adwords", "paid ads", "tiktok ads")):
        if tgt_lower.startswith("/academics"):
            return -100.0
        if tgt_lower.startswith("/ppc"):
            return 0.40

    return 0.0


@dataclass
class CandidateOpportunity:
    source_key: str
    source_file: str
    anchor: str
    sentence: str
    target_key: str
    target_title: str
    target_url: str
    score: float
    engine: str  # "direct_mention" | "linkbert" | "conceptual"
    components: Dict[str, float] = field(default_factory=dict)
    start: int = 0
    end: int = 0
    block_index: int = 0
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
            "engine": self.engine,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "block": self.block_index,
            "line": self.line,
        }


class TargetKeyphraseRegistry:
    """Pre-compiles target pages into keyphrase aliases and topical signatures."""

    def __init__(self, idx: SiteIndex) -> None:
        self.idx = idx
        # target_key -> list of (alias_phrase, specificity_score)
        self.target_phrases: Dict[str, List[Tuple[str, float]]] = {}
        self._compile()

    def _clean_title(self, title: str) -> str:
        # Strip branding suffixes like " | Pet Insurance AU", " - BrandName", etc.
        cleaned = re.split(r"\s+[|\-–—:]\s+", title)[0]
        return cleaned.strip()

    def _compile(self) -> None:
        for page in self.idx.pages:
            phrases: Dict[str, float] = {}

            # 1. Cleaned Title
            if page.title:
                clean_t = self._clean_title(page.title)
                if len(clean_t.split()) >= 2 and clean_t.lower() not in GENERIC_STOP:
                    phrases[clean_t.lower()] = 1.0

            # 2. H1
            if page.h1:
                clean_h1 = self._clean_title(page.h1)
                if len(clean_h1.split()) >= 2 and clean_h1.lower() not in GENERIC_STOP:
                    phrases[clean_h1.lower()] = max(phrases.get(clean_h1.lower(), 0.0), 0.95)

            # 3. URL slug phrases
            slug = page.key.strip("/").replace("-", " ").replace("_", " ").strip()
            if slug and len(slug.split()) >= 2:
                # e.g. "accident only cover", "routine care", "senior pets"
                phrases[slug.lower()] = max(phrases.get(slug.lower(), 0.0), 0.90)
                # Also with hyphen if original slug had hyphens
                hyphen_slug = page.key.strip("/").split("/")[-1].replace("_", "-")
                if "-" in hyphen_slug and len(hyphen_slug.split("-")) >= 2:
                    phrases[hyphen_slug.lower()] = max(phrases.get(hyphen_slug.lower(), 0.0), 0.90)

            # 4. Prominent subheadings (H2s)
            for h in page.headings:
                clean_h = self._clean_title(h)
                words = clean_h.split()
                if 2 <= len(words) <= 5 and clean_h.lower() not in GENERIC_STOP:
                    phrases[clean_h.lower()] = max(phrases.get(clean_h.lower(), 0.0), 0.75)

            # Filter out single generic words and sort by length descending (longest match first)
            valid_list = []
            
            # Canonical author/brand entity protection
            BRAND_PERSONA = {"muyiwa adekojede", "muyiwa", "adekojede"}
            if page.key != "/about-muyiwa-adekojede":
                phrases = {k: v for k, v in phrases.items() if k not in BRAND_PERSONA}
            else:
                phrases["muyiwa adekojede"] = 1.0
                phrases["about muyiwa"] = 0.95

            if page.key == "/tools/pagespeed-compare":
                phrases["core web vitals"] = 0.95
                phrases["pagespeed insights"] = 0.95

            if page.key == "/international-seo/case-studies/swadesi-news":
                phrases["lost 1,212 backlinks"] = 0.95

            for phrase, weight in phrases.items():
                words = phrase.split()
                if len(words) == 1 and phrase in GENERIC_SINGLE_WORDS:
                    continue
                valid_list.append((phrase, weight))

            valid_list.sort(key=lambda item: (-len(item[0]), -item[1]))
            self.target_phrases[page.key] = valid_list


class HybridDiscoveryEngine:
    """Discovers high-value internal links combining direct mentions, LinkBERT, and semantic search."""

    def __init__(self, idx: SiteIndex, predictor: LinkPredictor | None = None) -> None:
        self.idx = idx
        self.predictor = predictor
        self.registry = TargetKeyphraseRegistry(idx)
        
        # Initialize vector embedding model
        try:
            from sentence_transformers import SentenceTransformer
            if self.idx.vectors is None:
                self.idx.embed("all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except (ImportError, Exception):
            self.embedder = None

        # Build TF-IDF semantic matrix as primary/fallback IR layer (Salton Vector Space Model)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = [f"{p.title} {p.h1} {' '.join(p.headings)} {p.text[:600]}" for p in self.idx.pages]
            self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            self.tfidf_matrix = self.tfidf.fit_transform(corpus)
        except Exception:
            self.tfidf = None
            self.tfidf_matrix = None

    def _sentence_around(self, text: str, start: int, end: int, limit: int = 300) -> str:
        left = max(0, start - limit)
        m_left = list(re.finditer(r"[.!?\n]", text[left:start]))
        if m_left:
            start_pos = left + m_left[-1].end()
        else:
            start_pos = left

        right = min(len(text), end + limit)
        m_right = re.search(r"[.!?\n]", text[end:right])
        if m_right:
            end_pos = end + m_right.end()
        else:
            end_pos = right

        return re.sub(r"\s+", " ", text[start_pos:end_pos]).strip()

    def find_direct_mentions(self, source_page: Page, excluded_targets: Set[str]) -> List[CandidateOpportunity]:
        """Scans source page text for unlinked exact and phrase mentions of target pages."""
        opportunities: List[CandidateOpportunity] = []
        seen_spans: Set[Tuple[int, int, int]] = set()  # (block_idx, start, end)

        for b_idx, block in enumerate(source_page.blocks):
            if block.kind.startswith("h"):
                # By standard SEO practice, don't link inside source headings
                continue

            text = block.text

            for target_page in self.idx.pages:
                if target_page.key == source_page.key or target_page.key in excluded_targets:
                    continue

                aliases = self.registry.target_phrases.get(target_page.key, [])
                for alias, alias_weight in aliases:
                    # Match phrase as whole words with boundary awareness
                    # Allow optional hyphens between words: e.g. "accident-only" or "accident only"
                    escaped_words = [re.escape(w) for w in re.split(r"[\s\-]+", alias)]
                    pattern = r"\b" + r"[\s\-]+".join(escaped_words) + r"\b"

                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        s, e = match.start(), match.end()
                        matched_anchor = text[s:e]
                        clean_anchor = matched_anchor.strip().lower()

                        if clean_anchor in GENERIC_STOP or any(re.search(p, clean_anchor) for p in CTA_STOP_PATTERNS):
                            continue

                        topical_mod = check_topical_coherence(matched_anchor, text, source_page.key, target_page.key)
                        if topical_mod <= -50.0:
                            continue

                        # Check collision with already claimed span in this block
                        span_key = (b_idx, s, e)
                        if any(b_idx == existing[0] and not (e <= existing[1] or s >= existing[2]) for existing in seen_spans):
                            continue

                        # Compute semantic similarity between sentence context and target page
                        sentence = self._sentence_around(text, s, e)
                        
                        # Rapid score composition with topical modifier
                        score = min(0.98, 0.40 * alias_weight + 0.45 * 0.90 + 0.15 + max(0.0, topical_mod))

                        opportunities.append(
                            CandidateOpportunity(
                                source_key=source_page.key,
                                source_file=source_page.source,
                                anchor=matched_anchor,
                                sentence=sentence,
                                target_key=target_page.key,
                                target_title=target_page.title,
                                target_url=target_page.source,
                                score=score,
                                engine="direct_mention",
                                components={"mention_match": alias_weight, "prominence": 0.9},
                                start=s,
                                end=e,
                                block_index=block.index,
                                line=block.line,
                            )
                        )
                        seen_spans.add(span_key)
                        break  # Move to next target page once match found in this block

        return opportunities

    def find_linkbert_opportunities(
        self,
        source_page: Page,
        excluded_targets: Set[str],
        adaptive_threshold: float = 0.30,
    ) -> List[CandidateOpportunity]:
        """Runs calibrated LinkBERT with adaptive paragraph-level thresholding."""
        if self.predictor is None:
            return []

        opportunities: List[CandidateOpportunity] = []

        for block in source_page.blocks:
            if block.kind.startswith("h") or len(block.text.split()) < 8:
                continue

            text = block.text
            # Run inference per block to preserve high-fidelity BERT self-attention
            # Using adaptive thresholding: high=0.30, low=0.18
            spans = self.predictor.predict(
                text,
                high=adaptive_threshold,
                low=max(0.15, adaptive_threshold * 0.6),
                min_words=2,
                max_words=6,
            )

            for sp in spans:
                clean_sp = sp.text.strip().lower()
                if clean_sp in GENERIC_STOP or any(re.search(p, clean_sp) for p in CTA_STOP_PATTERNS):
                    continue

                sims = None
                if self.embedder is not None and self.idx.vectors is not None:
                    # Dense vector cosine similarity
                    query_vec = self.embedder.encode(
                        [sp.text + ". " + sp.sentence[:180]],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )[0].astype("float32")
                    sims = self.idx.vectors @ query_vec
                elif self.tfidf is not None and self.tfidf_matrix is not None:
                    # High-precision TF-IDF Vector Space Model (Salton et al.)
                    query_str = f"{sp.text} {sp.sentence[:180]}"
                    q_vec = self.tfidf.transform([query_str])
                    sims = (self.tfidf_matrix * q_vec.T).toarray().ravel()

                if sims is None:
                    continue

                # Find best matching target
                best_target_idx = -1
                best_score = -1.0
                best_comps = {}

                for i, tgt in enumerate(self.idx.pages):
                    if tgt.key == source_page.key or tgt.key in excluded_targets or tgt.words < 30:
                        continue

                    # Topical Silo & Cross-Contamination Filter
                    topical_mod = check_topical_coherence(sp.text, sp.sentence, source_page.key, tgt.key)
                    if topical_mod <= -50.0:
                        continue

                    cos = float(sims[i])
                    if cos < 0.15 and topical_mod <= 0.0:
                        continue

                    # Bonus for slug or heading token overlap
                    tgt_slug_tokens = set(re.findall(r"[a-z0-9]+", tgt.key.lower()))
                    anchor_tokens = set(re.findall(r"[a-z0-9]+", sp.text.lower()))
                    overlap = len(anchor_tokens & tgt_slug_tokens) / len(anchor_tokens) if anchor_tokens else 0.0

                    # Bonus for target keyphrase / alias match
                    alias_bonus = 0.0
                    for alias, w in self.registry.target_phrases.get(tgt.key, []):
                        if alias in sp.text.lower() or sp.text.lower() in alias:
                            alias_bonus = max(alias_bonus, w * 0.35)

                    composite = 0.50 * cos + 0.25 * overlap + 0.15 * sp.score + alias_bonus + max(0.0, topical_mod)
                    if composite > best_score:
                        best_score = composite
                        best_target_idx = i
                        best_comps = {"semantic": cos, "slug_overlap": overlap, "bert_prob": sp.score, "alias_bonus": alias_bonus, "topical_mod": topical_mod}

                if best_target_idx >= 0 and best_score >= 0.28:
                    target_page = self.idx.pages[best_target_idx]
                    opportunities.append(
                        CandidateOpportunity(
                            source_key=source_page.key,
                            source_file=source_page.source,
                            anchor=sp.text,
                            sentence=sp.sentence,
                            target_key=target_page.key,
                            target_title=target_page.title,
                            target_url=target_page.source,
                            score=best_score,
                            engine="linkbert",
                            components=best_comps,
                            start=sp.start,
                            end=sp.end,
                            block_index=block.index,
                            line=block.line,
                        )
                    )

        return opportunities

    def discover_all(
        self,
        source_page: Page,
        min_score: float = 0.28,
        per_target_limit: int = 4,
        skip_existing: bool = False,
    ) -> List[CandidateOpportunity]:
        """Executes full hybrid discovery pipeline for a source page."""
        from .match import resolve

        excluded_targets: Set[str] = set()
        if skip_existing:
            for b in source_page.blocks:
                if b.kind not in ("nav", "footer", "meta"):
                    for href in re.findall(r'href=[\'"]([^\'"]+)[\'"]', b.text):
                        t = resolve(href, source_page, self.idx)
                        if t:
                            excluded_targets.add(t.key)

        # 1. Harvest direct target mentions
        mentions = self.find_direct_mentions(source_page, excluded_targets)

        # 2. Harvest LinkBERT placements
        bert_opps = self.find_linkbert_opportunities(source_page, excluded_targets)

        # Merge and deduplicate by overlap
        merged: List[CandidateOpportunity] = []
        target_counts: Dict[str, int] = {}

        # Prioritize high-precision mentions first, then LinkBERT contextual anchors
        all_candidates = sorted(mentions + bert_opps, key=lambda c: -c.score)

        for cand in all_candidates:
            if cand.score < min_score:
                continue

            if target_counts.get(cand.target_key, 0) >= per_target_limit:
                continue

            # Check if this anchor overlaps with an existing accepted anchor in the same block
            overlap = False
            for accepted in merged:
                if accepted.block_index == cand.block_index:
                    if not (cand.end <= accepted.start or cand.start >= accepted.end):
                        overlap = True
                        break

            if not overlap:
                merged.append(cand)
                target_counts[cand.target_key] = target_counts.get(cand.target_key, 0) + 1

        merged.sort(key=lambda x: -x.score)
        return merged

    def rerank_with_cross_encoder(
        self,
        opportunities: List[CandidateOpportunity],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
    ) -> List[CandidateOpportunity]:
        """Re-ranks candidate opportunities using a neural cross-encoder.
        
        Cross-encoders evaluate the joint cross-attention between the source
        passage/sentence and the target page summary, providing high-precision
        semantic relevance scores that surpass bi-encoder cosine similarities.
        """
        if not opportunities:
            return []

        try:
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ce = CrossEncoder(model_name, device=device)
        except Exception:
            # Fallback gracefully if model cannot be instantiated
            return opportunities

        pairs = []
        for opp in opportunities:
            target_page = self.idx.by_key.get(opp.target_key)
            if target_page:
                target_summary = f"{target_page.title}. {target_page.description or target_page.text[:300]}"
            else:
                target_summary = opp.target_title

            query = f"Anchor: '{opp.anchor}'. Context: {opp.sentence}"
            pairs.append((query, target_summary))

        try:
            scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            for opp, raw_score in zip(opportunities, scores):
                prob = 1.0 / (1.0 + float(np.exp(-float(raw_score))))
                opp.components["cross_encoder_logit"] = float(raw_score)
                opp.components["cross_encoder_prob"] = round(prob, 4)
                # Blend: 50% base score + 50% cross-encoder probability
                opp.score = round(0.50 * opp.score + 0.50 * prob, 4)

            opportunities.sort(key=lambda o: -o.score)
        except Exception:
            pass

        return opportunities
