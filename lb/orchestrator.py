"""LinkGraph Pro — Agent Orchestration Protocol.

This module defines the exact operational procedures for an AI agent to execute
the full LinkGraph Pro pipeline using BrowserOS Neo and local codebase access.

It provides step-by-step runbooks that any AI agent in any IDE can follow
to perform comprehensive internal + external link optimization on any website.

Two operating modes:
  Mode A — Local Codebase: Agent has access to the project source files on disk.
  Mode B — URL Only: Agent has only a live URL and must use BrowserOS Neo to crawl.

The agent orchestrator reads this file and executes the protocol sequentially.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class OperatingMode(Enum):
    LOCAL_CODEBASE = "local"    # Agent has access to source files on disk
    URL_ONLY = "url"            # Agent has only a live URL


@dataclass
class ProjectConfig:
    """Configuration for a LinkGraph Pro analysis run."""
    target: str                     # folder path or URL
    mode: OperatingMode
    index_dir: str = ".linkbert"    # where to store index data
    
    # Discovery settings
    min_score: float = 0.35
    per_target_limit: int = 2
    max_pages: int = 200
    max_depth: int = 3
    
    # External linking
    external_targets_file: Optional[str] = None
    search_console_csv: Optional[str] = None
    
    # Apply settings
    auto_apply: bool = False
    dry_run: bool = True
    create_backups: bool = True
    
    # BrowserOS Neo
    use_browseros: bool = True
    site_url: Optional[str] = None  # canonical site URL for Search Console lookup


# ---------------------------------------------------------------------------
# The Agent Protocol (what any AI agent should follow)
# ---------------------------------------------------------------------------

AGENT_PROTOCOL = """
# LinkGraph Pro -- Agent Execution Protocol
# =========================================
#
# This protocol defines exactly how an AI agent should execute the full
# link optimization pipeline. Follow steps in order. Each step includes
# success criteria and failure handling.
#
# [!] This protocol requires BrowserOS Neo for browser-dependent steps.
#     If BrowserOS Neo is unavailable, skip browser steps and note limitations.
#
# =======================================================================
# PHASE 0: PROJECT DISCOVERY
# =======================================================================
#
# STEP 0.1: Determine Operating Mode
# -----------------------------------
# Input: User provides either a folder path or a URL
# Action:
#   - If input is a local folder path -> Mode A (LOCAL_CODEBASE)
#   - If input is a URL -> Mode B (URL_ONLY)
#   - If input is a folder AND user provides a URL -> Mode A + URL for live rendering
#
# STEP 0.2: Detect Project Framework (Mode A only)
# -------------------------------------------------
# Action: Examine project root for framework signatures:
#   - package.json -> check for next, nuxt, astro, gatsby, vuepress, docusaurus
#   - _config.yml -> Jekyll
#   - hugo.toml/config.toml -> Hugo
#   - mkdocs.yml -> MkDocs
#   - pyproject.toml/setup.py -> Python-based SSG
#   - Plain .html files -> Static HTML
# Purpose: Determines which AST parser to use for safe link injection
#
# STEP 0.3: Locate Build Output (Mode A only)
# --------------------------------------------
# Action: Find the static build directory. Check in order:
#   dist/, out/, build/, _site/, public/, .output/, site/, .next/
# If no build exists, offer to build the project or work with source files.
#
# =======================================================================
# PHASE 1: SITE INGESTION & INDEXING
# =======================================================================
#
# STEP 1.1: Scan and Index
# ------------------------
# Command: linkbert scan <target> --index <index_dir>
# Success: "Indexed N pages in X.Xs"
# Failure: "No pages found" -> check target path/URL, verify build exists
#
# STEP 1.2: Render Pages via BrowserOS Neo (if available)
# -------------------------------------------------------
# Action: For each indexed page (or top 50 by word count):
#   1. Open BrowserOS Neo tab
#   2. Navigate to page URL (local dev server or live URL)
#   3. Extract rendered accessibility tree, visible text blocks, bounding rects
#   4. Compare rendered content vs. static parse to find JS-only content
# Purpose: Catches content hidden behind client-side rendering
# Skip condition: Static HTML sites with no JavaScript
#
# =======================================================================
# PHASE 2: GRAPH ANALYSIS & TOPOLOGY
# =======================================================================
#
# STEP 2.1: Compute Site Graph Metrics
# -------------------------------------
# Command: linkbert graph --index <index_dir>
# Output: PageRank rankings, orphan pages, in/out degrees, clusters
# Alert user if: Orphan pages > 20% of total pages
#
# STEP 2.2: Detect Topic Clusters
# --------------------------------
# Action: Run cluster detection (URL hierarchy for <50 pages, Louvain for larger)
# Output: Named clusters with hub pages and cohesion scores
# Alert user if: Any cluster has cohesion < 0.30 (weak silo structure)
#
# =======================================================================
# PHASE 3: SEARCH CONSOLE INTEGRATION (Optional but Critical)
# =======================================================================
#
# STEP 3.1: Access Search Console via BrowserOS Neo
# --------------------------------------------------
# Prerequisites: User's Google account logged in via BrowserOS Neo
# Action:
#   1. Open BrowserOS Neo
#   2. Navigate to https://search.google.com/search-console/
#   3. Select the correct property (match against site URL)
#   4. Go to Performance -> Search Results
#   5. Set date range to Last 3 months
#   6. Export CSV (Queries + Pages tabs)
# Output: search_console_queries.csv, search_console_pages.csv
# [!] ASK USER: "I need to access Google Search Console for [site]. 
#               Please ensure you're logged into the correct Google account
#               in BrowserOS Neo. Should I proceed?"
#
# STEP 3.2: Detect Keyword Cannibalization
# -----------------------------------------
# Action: Parse Search Console CSV, identify queries with 2+ landing pages
# Output: Cannibalization report with consolidation recommendations
# Impact: Directly influences internal link anchor text selection
#
# =======================================================================
# PHASE 4: INTERNAL LINK DISCOVERY
# =======================================================================
#
# STEP 4.1: Run Hybrid Discovery Engine
# --------------------------------------
# Command: linkbert suggest --index <index_dir> --mode hybrid
# Output: Link opportunities with source, anchor, target, score, engine
# Validation: 
#   - Every orphan page should have at least 1 inbound link recommendation
#   - No page should link to itself
#   - No duplicate anchors to the same target from the same source
#
# STEP 4.2: Apply Silo Enforcement Guard
# ---------------------------------------
# Action: Check each recommendation against detected topic clusters
# Filter: Flag cross-silo links that don't go through hub pages
# Decision: Present flagged links to user with explanation
#
# STEP 4.3: Compute Anchor Diversity Metrics
# -------------------------------------------
# Action: Calculate Shannon entropy of anchor texts per target page
# Alert if: Any target has entropy < 1.5 bits or exact-match ratio > 60%
# Action: Suggest anchor text variations for over-optimized targets
#
# =======================================================================
# PHASE 5: EXTERNAL LINK DISCOVERY
# =======================================================================
#
# STEP 5.1: Extract Citable Claims and Entities
# ----------------------------------------------
# Command: linkbert external --index <index_dir>
# Output: Uncited statistics, authority entities, curated targets
#
# STEP 5.2: Validate External Targets via BrowserOS Neo
# -----------------------------------------------------
# Action: For each recommended external URL:
#   1. Open in BrowserOS Neo
#   2. Verify HTTP 200, no paywall/login wall, content relevance
#   3. Check rel="nofollow" policy
#   4. Verify canonical URL (no redirect chains)
# Filter: Remove any URL returning 4xx/5xx or behind paywall
#
# STEP 5.3: SERP Competitor Guard (via BrowserOS Neo)
# ---------------------------------------------------
# Action: For each external domain being recommended:
#   1. Take the site's top 10 Search Console queries
#   2. Search each query in Google via BrowserOS Neo
#   3. Check if the external domain appears in top 20 results
#   4. REJECT the external link if the domain competes for the same query
# Purpose: Prevents bleeding ranking equity to a direct SERP competitor
# [!] This step requires BrowserOS Neo with Google access
#
# =======================================================================
# PHASE 6: RECOMMENDATION & APPLICATION
# =======================================================================
#
# STEP 6.1: Present Consolidated Report
# --------------------------------------
# Output to user:
#   a) Summary statistics (total opportunities, orphans rescued, equity gain)
#   b) Internal link table: source -> anchor -> target with scores
#   c) External link table: source -> anchor -> external URL with category
#   d) Cannibalization alerts (if Search Console data available)
#   e) Silo violation warnings
#   f) Anchor diversity alerts
#   g) Interactive dashboard at .linkbert/report.html
#
# STEP 6.2: Apply Links (Mode A - Local Codebase)
# ------------------------------------------------
# [!] ASK USER: "I have N link recommendations ready. 
#               Should I apply them to your source files?
#               I'll create .bak backups of every modified file."
#
# If approved:
#   Command: linkbert apply --index <index_dir> --min-score 0.40
#   Verify: Each modified file still compiles/renders correctly
#   Offer: linkbert rollback if anything looks wrong
#
# STEP 6.3: Deliver Recommendations (Mode B - URL Only)
# -----------------------------------------------------
# Output: Structured report with exact page URLs, text excerpts showing
#         precisely where each link should be placed, and the HTML snippet
#         to insert. User must manually implement changes.
#
# =======================================================================
# PHASE 7: VERIFICATION & VALIDATION
# =======================================================================
#
# STEP 7.1: Graph Invariant Verification
# ---------------------------------------
# Check: Zero orphan pages remaining (all reachable from root in <=3 hops)
# Check: PageRank convergence delta < 1e-6
# Check: No self-links introduced
# Check: No duplicate links (same source->target) introduced
#
# STEP 7.2: Link Health Verification
# -----------------------------------
# Action: HTTP HEAD request to every internal and external link target
# Report: Any 4xx/5xx status codes or redirect chains > 2 hops
#
# STEP 7.3: Build Compilation Test (Mode A)
# -----------------------------------------
# Action: Run the project's build command to verify no markup corruption
# Check: Zero build errors, zero HTML validation warnings
#
# =======================================================================
# END OF PROTOCOL
# =======================================================================
"""


def get_protocol() -> str:
    """Returns the full agent execution protocol as a string."""
    return AGENT_PROTOCOL


def save_protocol(out_dir: Path) -> Path:
    """Saves the protocol to a file for reference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "AGENT_PROTOCOL.md"
    path.write_text(AGENT_PROTOCOL, encoding="utf-8")
    return path
