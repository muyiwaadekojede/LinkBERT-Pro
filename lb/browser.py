"""BrowserOS Neo integration for rendered DOM extraction, SERP queries, and Search Console access.

This module provides the bridge between LinkGraph Pro and BrowserOS Neo, enabling:
1. Rendered Page Extraction: Post-JS-hydration accessibility tree and visible text
2. Google Search Console Data: Query performance matrix (queries, clicks, impressions, position)
3. Live SERP Verification: Google/Bing search for external link target validation
4. HTTP Link Health Checks: Verify all recommended links resolve to 200 OK

BrowserOS Neo is a real browser with persistent user sessions and live logins.
When BrowserOS Neo is unavailable, functions degrade gracefully to headless/static methods.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, quote_plus

import requests


# ---------------------------------------------------------------------------
# Data structures for browser-extracted content
# ---------------------------------------------------------------------------

@dataclass
class RenderedBlock:
    """A visible text block extracted from the browser-rendered DOM."""
    text: str
    tag: str           # p, li, span, div, etc.
    selector: str      # CSS selector path
    index: int
    is_visible: bool
    viewport_y: float  # pixels from top of document
    font_size: float   # computed px
    in_main_content: bool  # heuristic: inside <main>, <article>, or primary content area
    line: int | None = None


@dataclass
class RenderedPage:
    """Full rendered page data extracted via BrowserOS Neo."""
    url: str
    title: str
    description: str
    h1: str
    headings: List[str]
    blocks: List[RenderedBlock]
    internal_links: List[Dict[str, str]]  # [{href, anchor_text, is_nav}]
    external_links: List[Dict[str, str]]  # [{href, anchor_text, rel}]
    schema_entities: List[Dict[str, Any]]  # extracted JSON-LD / microdata
    word_count: int
    render_method: str  # "browseros_neo" | "static_parse"


@dataclass
class SearchConsoleRow:
    """One row from Google Search Console Performance report."""
    query: str
    page: str
    clicks: int
    impressions: int
    ctr: float
    position: float


@dataclass
class SERPResult:
    """One organic search result from a SERP query."""
    title: str
    url: str
    domain: str
    position: int
    snippet: str


# ---------------------------------------------------------------------------
# BrowserOS Neo availability check
# ---------------------------------------------------------------------------

def check_browseros_available() -> bool:
    """Check if BrowserOS Neo MCP tools are accessible in the current environment."""
    # This is a runtime check - BrowserOS Neo tools are MCP-based
    # and available when the plugin is connected.
    # The actual availability is determined by the agent runtime.
    return True  # Placeholder; actual check happens at MCP call time


# ---------------------------------------------------------------------------
# SERP Query Functions (used by the agent via BrowserOS Neo)
# ---------------------------------------------------------------------------

def build_google_search_url(query: str, num_results: int = 20) -> str:
    """Builds a Google Search URL for the given query."""
    return f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"


def build_bing_search_url(query: str, count: int = 20) -> str:
    """Builds a Bing Search URL for the given query."""
    return f"https://www.bing.com/search?q={quote_plus(query)}&count={count}"


def build_search_console_url(site_url: str) -> str:
    """Builds the Google Search Console Performance URL for a property."""
    encoded = quote_plus(site_url)
    return f"https://search.google.com/search-console/performance/search-analytics?resource_id={encoded}"


# ---------------------------------------------------------------------------
# Static HTTP fallback functions (when BrowserOS Neo is unavailable)
# ---------------------------------------------------------------------------

def verify_link_health(urls: List[str], timeout: int = 10) -> List[Dict[str, Any]]:
    """Verify HTTP status of a list of URLs using HEAD requests.
    
    Returns list of {url, status_code, redirect_target, is_healthy, error}
    """
    results = []
    for url in urls:
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True, 
                            headers={"User-Agent": "LinkGraphPro/2.0 (link-health-check)"})
            results.append({
                "url": url,
                "status_code": r.status_code,
                "redirect_target": r.url if r.url != url else None,
                "is_healthy": r.status_code in {200, 301, 302},
                "error": None,
            })
        except requests.RequestException as e:
            results.append({
                "url": url,
                "status_code": None,
                "redirect_target": None,
                "is_healthy": False,
                "error": str(e),
            })
    return results


def extract_domain(url: str) -> str:
    """Extract the root domain from a URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    # Strip www.
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


# ---------------------------------------------------------------------------
# Competitor SERP Guard
# ---------------------------------------------------------------------------

@dataclass
class SERPConflict:
    """Represents a conflict where an external link target competes with the site."""
    external_url: str
    external_domain: str
    competing_query: str
    external_position: int
    site_position: int | None
    risk_level: str  # "high" | "medium" | "low"


def check_serp_conflicts(
    external_domains: Set[str],
    site_queries: List[SearchConsoleRow],
    serp_results: Dict[str, List[SERPResult]],
) -> List[SERPConflict]:
    """Check if any recommended external link domains compete in SERPs with the site.
    
    Args:
        external_domains: Set of domains we're considering linking to
        site_queries: Search Console data showing what queries the site ranks for
        serp_results: Pre-fetched SERP results for the site's top queries
    
    Returns:
        List of conflicts where external domains rank for the site's queries
    """
    conflicts = []
    
    for query_row in site_queries:
        query = query_row.query
        if query not in serp_results:
            continue
            
        for result in serp_results[query]:
            result_domain = extract_domain(result.url)
            if result_domain in external_domains and result.position <= 20:
                risk = "high" if result.position <= 5 else ("medium" if result.position <= 10 else "low")
                conflicts.append(SERPConflict(
                    external_url=result.url,
                    external_domain=result_domain,
                    competing_query=query,
                    external_position=result.position,
                    site_position=round(query_row.position) if query_row.position else None,
                    risk_level=risk,
                ))
    
    return conflicts


# ---------------------------------------------------------------------------
# Search Console Data Processing
# ---------------------------------------------------------------------------

def parse_search_console_csv(csv_text: str) -> List[SearchConsoleRow]:
    """Parse a Search Console Performance CSV export into structured rows."""
    import csv
    from io import StringIO
    
    rows = []
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        try:
            rows.append(SearchConsoleRow(
                query=row.get("Top queries", row.get("Query", row.get("query", ""))),
                page=row.get("Top pages", row.get("Page", row.get("page", ""))),
                clicks=int(float(row.get("Clicks", row.get("clicks", 0)))),
                impressions=int(float(row.get("Impressions", row.get("impressions", 0)))),
                ctr=float(row.get("CTR", row.get("ctr", "0")).replace("%", "")) / 100 
                    if "%" in str(row.get("CTR", row.get("ctr", "0"))) 
                    else float(row.get("CTR", row.get("ctr", 0))),
                position=float(row.get("Position", row.get("position", 0))),
            ))
        except (ValueError, KeyError):
            continue
    return rows


def build_cannibalization_matrix(
    rows: List[SearchConsoleRow],
) -> Dict[str, List[Dict[str, Any]]]:
    """Detect keyword cannibalization: queries where multiple pages compete.
    
    Returns: {query: [{page, clicks, impressions, position}, ...]} 
             for queries with 2+ pages ranking.
    """
    query_pages: Dict[str, List[SearchConsoleRow]] = {}
    for row in rows:
        query_pages.setdefault(row.query, []).append(row)
    
    conflicts = {}
    for query, pages in query_pages.items():
        if len(pages) >= 2:
            # Sort by position (best first)
            pages_sorted = sorted(pages, key=lambda r: r.position)
            conflicts[query] = [
                {
                    "page": p.page,
                    "clicks": p.clicks,
                    "impressions": p.impressions,
                    "position": round(p.position, 1),
                }
                for p in pages_sorted
            ]
    
    return conflicts


# ---------------------------------------------------------------------------
# Instructions for Agent Orchestration
# ---------------------------------------------------------------------------

# These constants define the BrowserOS Neo scripts the agent should use
# when operating in browser mode. They are reference templates.

BROWSEROS_EXTRACT_PAGE_SCRIPT = """
// BrowserOS Neo run script: Extract rendered page content
// Use inside a `run` call after navigating to the target URL

const page = await sdk.activePage();
const snapshot = await page.snapshot();

// Extract all visible text blocks with their computed properties
const blocks = await page.evaluate(() => {
    const blockTags = new Set(['P', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                                'BLOCKQUOTE', 'TD', 'TH', 'FIGCAPTION', 'DD', 'DT']);
    const dropTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'SVG', 'IFRAME', 
                               'NAV', 'FOOTER', 'HEADER', 'ASIDE', 'FORM', 'BUTTON']);
    
    const results = [];
    const walker = document.createTreeWalker(
        document.body || document.documentElement,
        NodeFilter.SHOW_ELEMENT,
        {
            acceptNode(node) {
                if (dropTags.has(node.tagName)) return NodeFilter.FILTER_REJECT;
                if (blockTags.has(node.tagName)) return NodeFilter.FILTER_ACCEPT;
                return NodeFilter.FILTER_SKIP;
            }
        }
    );
    
    let idx = 0;
    let node;
    while (node = walker.nextNode()) {
        const text = node.innerText?.trim();
        if (!text || text.length < 12) continue;
        
        const rect = node.getBoundingClientRect();
        const style = getComputedStyle(node);
        const isVisible = rect.width > 0 && rect.height > 0 && 
                         style.display !== 'none' && style.visibility !== 'hidden';
        
        if (!isVisible) continue;
        
        // Check if inside main content area
        const inMain = !!node.closest('main, article, [role="main"], .content, .post, .entry');
        
        results.push({
            text: text.replace(/\\s+/g, ' '),
            tag: node.tagName.toLowerCase(),
            index: idx++,
            is_visible: true,
            viewport_y: rect.top + window.scrollY,
            font_size: parseFloat(style.fontSize),
            in_main_content: inMain,
        });
    }
    return results;
});

// Extract all links with context
const links = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('a[href]')).map(a => {
        const isNav = !!a.closest('nav, header, footer, [role="navigation"]');
        const href = a.href;
        const isExternal = href && !href.startsWith(window.location.origin);
        return {
            href: href,
            anchor_text: (a.innerText || '').trim().substring(0, 200),
            is_nav: isNav,
            is_external: isExternal,
            rel: a.getAttribute('rel') || '',
        };
    }).filter(l => l.href && !l.href.startsWith('javascript:') && !l.href.startsWith('mailto:'));
});

// Extract JSON-LD structured data
const jsonLd = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('script[type="application/ld+json"]')).map(s => {
        try { return JSON.parse(s.textContent); } catch { return null; }
    }).filter(Boolean);
});

return { blocks, links, jsonLd, title: document.title, url: window.location.href };
"""

BROWSEROS_SEARCH_CONSOLE_SCRIPT = """
// BrowserOS Neo run script: Extract Search Console Performance data
// Prerequisites: User must be logged into Google account with Search Console access

const page = await sdk.activePage();

// Navigate to Search Console
await page.navigate('{search_console_url}');
await page.waitForSelector('[data-view="performance"]', { timeout: 15000 });

// Wait for data to load
await page.waitForText('Total clicks', { timeout: 10000 });

// Click export button and download CSV
const exportBtn = await page.querySelector('[aria-label="Export"]');
if (exportBtn) {
    await exportBtn.click();
    await page.waitForText('Download CSV', { timeout: 5000 });
    const csvLink = await page.querySelector('a[download*="csv"], [data-type="csv"]');
    if (csvLink) await csvLink.click();
}

// Alternative: scrape the table directly
const tableData = await page.evaluate(() => {
    const rows = document.querySelectorAll('table tbody tr');
    return Array.from(rows).map(row => {
        const cells = row.querySelectorAll('td');
        return {
            query: cells[0]?.innerText?.trim() || '',
            clicks: parseInt(cells[1]?.innerText?.replace(/,/g, '') || '0'),
            impressions: parseInt(cells[2]?.innerText?.replace(/,/g, '') || '0'),
            ctr: cells[3]?.innerText?.trim() || '0%',
            position: parseFloat(cells[4]?.innerText || '0'),
        };
    });
});

return tableData;
"""
