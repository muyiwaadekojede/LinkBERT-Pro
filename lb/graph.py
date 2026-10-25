"""Topological graph analysis, PageRank simulation, and link equity metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Set, Tuple

from .site import Page, SiteIndex


@dataclass
class GraphMetrics:
    page_rank: Dict[str, float]
    projected_page_rank: Dict[str, float]
    in_degrees: Dict[str, int]
    out_degrees: Dict[str, int]
    orphan_pages: List[str]
    under_linked_pages: List[str]
    clusters: Dict[str, List[str]]
    anchor_distributions: Dict[str, Dict[str, int]]


class LinkGraph:
    """Simulates internal PageRank and link equity flow across the site."""

    def __init__(self, idx: SiteIndex, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.idx = idx
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.pages = idx.pages
        self.n = len(self.pages)
        self.key_to_idx = {p.key: i for i, p in enumerate(self.pages)}

    def build_adjacency(self, additional_links: List[Tuple[str, str]] | None = None) -> np.ndarray:
        """Builds directed adjacency matrix A where A[i, j] = 1 if page i links to page j."""
        adj = np.zeros((self.n, self.n), dtype=np.float32)

        from .match import resolve

        for i, page in enumerate(self.pages):
            for href in page.links:
                target = resolve(href, page, self.idx)
                if target and target.key in self.key_to_idx and target.key != page.key:
                    j = self.key_to_idx[target.key]
                    adj[i, j] = 1.0

        if additional_links:
            for src_key, tgt_key in additional_links:
                if src_key in self.key_to_idx and tgt_key in self.key_to_idx and src_key != tgt_key:
                    i = self.key_to_idx[src_key]
                    j = self.key_to_idx[tgt_key]
                    adj[i, j] = 1.0

        return adj

    def compute_pagerank(self, adj: np.ndarray) -> np.ndarray:
        """Computes Google PageRank using power iteration."""
        if self.n == 0:
            return np.array([])
        if self.n == 1:
            return np.array([1.0])

        out_degrees = adj.sum(axis=1)
        # Stochastic transition matrix M where M[j, i] is probability of transitioning from i to j
        M = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            if out_degrees[i] > 0:
                M[:, i] = adj[i, :] / out_degrees[i]
            else:
                # Dangling node transitions uniformly to all pages
                M[:, i] = 1.0 / self.n

        # Power iteration
        pr = np.full(self.n, 1.0 / self.n, dtype=np.float32)
        teleport = (1.0 - self.damping) / self.n

        for _ in range(self.max_iter):
            new_pr = self.damping * (M @ pr) + teleport
            if np.sum(np.abs(new_pr - pr)) < self.tol:
                pr = new_pr
                break
            pr = new_pr

        # Normalize so mean PageRank = 1.0 for intuitive comparison
        if pr.sum() > 0:
            pr = (pr / pr.sum()) * self.n
        return pr

    def detect_clusters(self) -> Dict[str, List[str]]:
        """Groups pages into semantic and path-based topic clusters (silos)."""
        clusters: Dict[str, List[str]] = {}
        for p in self.pages:
            parts = [part for part in p.key.strip("/").split("/") if part]
            if len(parts) >= 1:
                cluster_name = f"/{parts[0]}/"
            else:
                cluster_name = "/"
            clusters.setdefault(cluster_name, []).append(p.key)
        return clusters

    def analyze(self, additional_links: List[Tuple[str, str]] | None = None) -> GraphMetrics:
        """Runs complete graph analysis including baseline and projected PageRank."""
        base_adj = self.build_adjacency()
        base_pr = self.compute_pagerank(base_adj)

        proj_adj = self.build_adjacency(additional_links) if additional_links else base_adj
        proj_pr = self.compute_pagerank(proj_adj)

        in_degrees = {p.key: int(base_adj[:, i].sum()) for i, p in enumerate(self.pages)}
        out_degrees = {p.key: int(base_adj[i, :].sum()) for i, p in enumerate(self.pages)}

        orphans = [p.key for p in self.pages if in_degrees[p.key] == 0 and p.key != "/"]
        under_linked = [p.key for p in self.pages if in_degrees[p.key] == 1 and p.key != "/"]

        # Sort by lowest in-degree then lowest PageRank
        orphans.sort(key=lambda k: base_pr[self.key_to_idx[k]])
        under_linked.sort(key=lambda k: base_pr[self.key_to_idx[k]])

        pr_dict = {p.key: round(float(base_pr[i]), 4) for i, p in enumerate(self.pages)}
        proj_pr_dict = {p.key: round(float(proj_pr[i]), 4) for i, p in enumerate(self.pages)}

        # Track existing anchor distribution per target
        anchor_dist: Dict[str, Dict[str, int]] = {p.key: {} for p in self.pages}

        return GraphMetrics(
            page_rank=pr_dict,
            projected_page_rank=proj_pr_dict,
            in_degrees=in_degrees,
            out_degrees=out_degrees,
            orphan_pages=orphans,
            under_linked_pages=under_linked,
            clusters=self.detect_clusters(),
            anchor_distributions=anchor_dist,
        )
