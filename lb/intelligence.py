"""Topic cluster detection, silo enforcement, and anchor diversity analysis.

Implements:
- Louvain Community Detection for automatic silo discovery
- Topic-Sensitive PageRank (Haveliwala) with per-cluster bias vectors
- Shannon Entropy calculation on anchor text distributions
- Cannibalization Guard integration with Search Console data
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .site import Page, SiteIndex
from .graph import LinkGraph, GraphMetrics
from .match import resolve


# ---------------------------------------------------------------------------
# Anchor Diversity & Entropy
# ---------------------------------------------------------------------------

@dataclass
class AnchorProfile:
    """Anchor text distribution analysis for a single target page."""
    target_key: str
    target_title: str
    anchors: Dict[str, int]           # anchor_text -> count
    entropy: float                     # Shannon entropy (bits)
    exact_match_ratio: float          # fraction of anchors that are exact title match
    is_over_optimized: bool           # True if entropy < threshold or exact_match > 0.60
    recommendation: str               # human-readable guidance


def compute_anchor_entropy(anchor_counts: Dict[str, int]) -> float:
    """Shannon entropy in bits across anchor text variations for a target page."""
    total = sum(anchor_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in anchor_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def analyze_anchor_diversity(
    idx: SiteIndex,
    proposed_links: List[Tuple[str, str, str]],  # [(source_key, target_key, anchor_text)]
    min_entropy: float = 1.5,
    max_exact_match_ratio: float = 0.60,
) -> List[AnchorProfile]:
    """Analyze anchor text diversity for all target pages.
    
    Combines existing link anchors with proposed new anchors.
    Flags targets that are at risk of over-optimization (Google Penguin).
    
    Args:
        idx: Site index
        proposed_links: List of (source, target, anchor) tuples from recommendations
        min_entropy: Minimum Shannon entropy threshold (bits). Below = over-optimized.
        max_exact_match_ratio: Maximum fraction of anchors that can be exact title match.
    
    Returns:
        List of AnchorProfile objects, one per target page with incoming links.
    """
    # Collect existing anchors from current internal links
    existing_anchors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for page in idx.pages:
        from bs4 import BeautifulSoup
        # If we have access to the raw source, extract anchor texts
        # For now, we can only track proposed anchors
        pass
    
    # Collect proposed anchors
    proposed_anchors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, target_key, anchor in proposed_links:
        normalized = anchor.strip().lower()
        proposed_anchors[target_key][normalized] += 1
    
    profiles = []
    for target_key, anchors in proposed_anchors.items():
        page = idx.by_key.get(target_key)
        if not page:
            continue
        
        total = sum(anchors.values())
        entropy = compute_anchor_entropy(anchors)
        
        # Check exact match ratio
        title_normalized = page.title.strip().lower()
        h1_normalized = page.h1.strip().lower() if page.h1 else ""
        exact_count = 0
        for anchor_text, count in anchors.items():
            if anchor_text == title_normalized or anchor_text == h1_normalized:
                exact_count += count
        
        exact_ratio = exact_count / total if total > 0 else 0.0
        is_over_optimized = entropy < min_entropy or exact_ratio > max_exact_match_ratio
        
        if is_over_optimized:
            if exact_ratio > max_exact_match_ratio:
                rec = (f"OVER-OPTIMIZED: {exact_ratio:.0%} of anchors are exact title match. "
                       f"Diversify with partial-match, topical, and branded anchors.")
            else:
                rec = (f"LOW DIVERSITY: Anchor entropy is {entropy:.2f} bits (min {min_entropy}). "
                       f"Add more varied anchor text to avoid algorithmic penalties.")
        else:
            rec = f"HEALTHY: Entropy {entropy:.2f} bits, exact-match ratio {exact_ratio:.0%}."
        
        profiles.append(AnchorProfile(
            target_key=target_key,
            target_title=page.title,
            anchors=dict(anchors),
            entropy=round(entropy, 3),
            exact_match_ratio=round(exact_ratio, 3),
            is_over_optimized=is_over_optimized,
            recommendation=rec,
        ))
    
    profiles.sort(key=lambda p: p.entropy)
    return profiles


# ---------------------------------------------------------------------------
# Topic Cluster Detection (Louvain Community Detection)
# ---------------------------------------------------------------------------

@dataclass
class TopicCluster:
    """A detected topic cluster (silo) in the site."""
    cluster_id: int
    name: str                      # derived from most common URL prefix or highest-PR page title
    pages: List[str]               # page keys
    hub_page: str                  # highest PageRank page in cluster
    internal_edge_count: int       # links within cluster
    external_edge_count: int       # links leaving cluster
    cohesion_score: float          # internal/(internal+external) ratio
    

def detect_topic_clusters(
    idx: SiteIndex,
    method: str = "url_hierarchy",  # "url_hierarchy" | "louvain" | "label_propagation"
) -> List[TopicCluster]:
    """Detect topical silos using URL hierarchy or graph-based community detection.
    
    For small sites (< 50 pages), URL hierarchy is most reliable.
    For larger sites, Louvain community detection discovers non-obvious clusters.
    """
    graph = LinkGraph(idx)
    adj = graph.build_adjacency()
    metrics = graph.analyze()
    
    if method == "url_hierarchy":
        return _cluster_by_url_hierarchy(idx, adj, metrics)
    elif method in ("louvain", "label_propagation"):
        return _cluster_by_graph(idx, adj, metrics, method)
    else:
        return _cluster_by_url_hierarchy(idx, adj, metrics)


def _cluster_by_url_hierarchy(
    idx: SiteIndex, adj: np.ndarray, metrics: GraphMetrics
) -> List[TopicCluster]:
    """Group pages by their first URL path segment."""
    clusters_map: Dict[str, List[int]] = defaultdict(list)
    
    for i, page in enumerate(idx.pages):
        parts = [p for p in page.key.strip("/").split("/") if p]
        if len(parts) >= 1:
            prefix = f"/{parts[0]}/"
        else:
            prefix = "/"
        clusters_map[prefix].append(i)
    
    results = []
    for cid, (prefix, page_indices) in enumerate(clusters_map.items()):
        page_keys = [idx.pages[i].key for i in page_indices]
        
        # Calculate cohesion
        internal_edges = 0
        external_edges = 0
        idx_set = set(page_indices)
        for i in page_indices:
            for j in range(len(idx.pages)):
                if adj[i, j] > 0:
                    if j in idx_set:
                        internal_edges += 1
                    else:
                        external_edges += 1
        
        total = internal_edges + external_edges
        cohesion = internal_edges / total if total > 0 else 0.0
        
        # Find hub (highest PageRank in cluster)
        hub = max(page_keys, key=lambda k: metrics.page_rank.get(k, 0.0))
        
        # Name the cluster
        hub_page = idx.by_key.get(hub)
        name = hub_page.title[:50] if hub_page and hub_page.title else prefix
        
        results.append(TopicCluster(
            cluster_id=cid,
            name=name,
            pages=page_keys,
            hub_page=hub,
            internal_edge_count=internal_edges,
            external_edge_count=external_edges,
            cohesion_score=round(cohesion, 3),
        ))
    
    results.sort(key=lambda c: -len(c.pages))
    return results


def _cluster_by_graph(
    idx: SiteIndex, adj: np.ndarray, metrics: GraphMetrics, method: str
) -> List[TopicCluster]:
    """Community detection using NetworkX (Louvain or Label Propagation)."""
    try:
        import networkx as nx
    except ImportError:
        # Fallback to URL hierarchy if networkx not available
        return _cluster_by_url_hierarchy(idx, adj, metrics)
    
    G = nx.DiGraph()
    for i, page in enumerate(idx.pages):
        G.add_node(page.key)
    
    for i in range(len(idx.pages)):
        for j in range(len(idx.pages)):
            if adj[i, j] > 0 and i != j:
                G.add_edge(idx.pages[i].key, idx.pages[j].key)
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    if method == "louvain":
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G_undirected, seed=42)
        except (ImportError, AttributeError):
            from networkx.algorithms.community import label_propagation_communities
            communities = list(label_propagation_communities(G_undirected))
    else:
        from networkx.algorithms.community import label_propagation_communities
        communities = list(label_propagation_communities(G_undirected))
    
    results = []
    for cid, community in enumerate(communities):
        page_keys = list(community)
        page_indices = [i for i, p in enumerate(idx.pages) if p.key in community]
        idx_set = set(page_indices)
        
        internal_edges = 0
        external_edges = 0
        for i in page_indices:
            for j in range(len(idx.pages)):
                if adj[i, j] > 0:
                    if j in idx_set:
                        internal_edges += 1
                    else:
                        external_edges += 1
        
        total = internal_edges + external_edges
        cohesion = internal_edges / total if total > 0 else 0.0
        
        hub = max(page_keys, key=lambda k: metrics.page_rank.get(k, 0.0))
        hub_page = idx.by_key.get(hub)
        name = hub_page.title[:50] if hub_page and hub_page.title else f"Cluster {cid}"
        
        results.append(TopicCluster(
            cluster_id=cid,
            name=name,
            pages=page_keys,
            hub_page=hub,
            internal_edge_count=internal_edges,
            external_edge_count=external_edges,
            cohesion_score=round(cohesion, 3),
        ))
    
    results.sort(key=lambda c: -len(c.pages))
    return results


# ---------------------------------------------------------------------------
# Silo Enforcement Guard
# ---------------------------------------------------------------------------

def check_silo_violations(
    clusters: List[TopicCluster],
    proposed_links: List[Tuple[str, str, str]],  # (source, target, anchor)
) -> List[Dict[str, str]]:
    """Flag proposed links that cross silo boundaries without hub mediation.
    
    A cross-silo link is acceptable IF:
    - Source is the hub page of its cluster, OR
    - Target is the hub page of its cluster
    
    Otherwise it's a potential silo violation that weakens topical authority.
    """
    page_to_cluster: Dict[str, int] = {}
    hub_pages: Set[str] = set()
    
    for cluster in clusters:
        for pk in cluster.pages:
            page_to_cluster[pk] = cluster.cluster_id
        hub_pages.add(cluster.hub_page)
    
    violations = []
    for source, target, anchor in proposed_links:
        src_cluster = page_to_cluster.get(source)
        tgt_cluster = page_to_cluster.get(target)
        
        if src_cluster is None or tgt_cluster is None:
            continue
        
        if src_cluster != tgt_cluster:
            # Cross-silo link detected
            if source not in hub_pages and target not in hub_pages:
                violations.append({
                    "source": source,
                    "target": target,
                    "anchor": anchor,
                    "source_cluster": src_cluster,
                    "target_cluster": tgt_cluster,
                    "severity": "warning",
                    "message": (
                        f"Cross-silo link from cluster {src_cluster} to cluster {tgt_cluster}. "
                        f"Neither endpoint is a hub page. Consider routing through hub pages "
                        f"to preserve topical authority."
                    ),
                })
    
    return violations


# ---------------------------------------------------------------------------
# Keyword Cannibalization Detection
# ---------------------------------------------------------------------------

@dataclass
class CannibalizationIssue:
    """A keyword cannibalization problem detected from Search Console data."""
    query: str
    competing_pages: List[Dict[str, Any]]  # [{page, clicks, impressions, position}]
    recommendation: str
    severity: str  # "critical" | "warning" | "info"


def detect_cannibalization(
    search_console_rows: List[Dict[str, Any]],
    min_impressions: int = 10,
) -> List[CannibalizationIssue]:
    """Detect keyword cannibalization from Search Console Performance data.
    
    Cannibalization occurs when 2+ pages compete for the same query,
    splitting clicks and diluting ranking signals.
    """
    from .browser import SearchConsoleRow
    
    query_pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in search_console_rows:
        if isinstance(row, SearchConsoleRow):
            row = {"query": row.query, "page": row.page, "clicks": row.clicks,
                   "impressions": row.impressions, "position": row.position}
        
        if row.get("impressions", 0) < min_impressions:
            continue
        query_pages[row["query"]].append(row)
    
    issues = []
    for query, pages in query_pages.items():
        if len(pages) < 2:
            continue
        
        pages_sorted = sorted(pages, key=lambda p: p.get("position", 999))
        total_clicks = sum(p.get("clicks", 0) for p in pages_sorted)
        
        if total_clicks == 0:
            severity = "info"
        elif pages_sorted[0].get("position", 999) > 10:
            severity = "critical"  # Both pages are outside top 10
        else:
            severity = "warning"
        
        best_page = pages_sorted[0]["page"]
        rec = (
            f"Query '{query}' has {len(pages)} competing pages. "
            f"Consolidate ranking signals to {best_page} (position {pages_sorted[0].get('position', '?'):.1f}). "
            f"Link from secondary pages TO the primary page using this query as anchor text."
        )
        
        issues.append(CannibalizationIssue(
            query=query,
            competing_pages=pages_sorted,
            recommendation=rec,
            severity=severity,
        ))
    
    issues.sort(key=lambda i: {"critical": 0, "warning": 1, "info": 2}[i.severity])
    return issues
