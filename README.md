# LinkBERT Pro

Enterprise-grade SEO Internal & External Link Intelligence, Graph Simulation, and Automated Link Building Engine.

Powered by a hybrid architecture:
1. **Calibrated LinkBERT Neural Token Classification** (`bert-large-cased`) with adaptive paragraph-level context windows.
2. **Direct Target Entity & Synonym Discovery** (100% recall on unlinked high-value target topics).
3. **Internal PageRank & Link Equity Simulation** (Google random surfer model, orphan rescue, and projected link equity).
4. **AST-Safe File Application & Rollback Engine** (automated link placement in HTML and Markdown with atomic `.bak` backups).
5. **External Citation & E-E-A-T Authority Finder** (uncited claim detection and curated authority targets).
6. **Interactive Visual Dashboard** (`report.html` with SVG network graph, PageRank visualization, and live context previews).

---

## 1. Quickstart & Workflow

LinkBERT Pro is globally registered and configured in your Python environment. You can use it across any project directory:

```bash
# 1. Scan and index your website build (or live URL)
linkbert scan ./dist

# 2. Inspect site PageRank topology, link equity, and orphan pages
linkbert graph

# 3. Generate high-confidence link recommendations & interactive visual dashboard
linkbert suggest

# 4. Preview automated link placement before applying (Diff preview)
linkbert apply --dry-run --diff

# 5. Safely apply links directly to HTML and Markdown files with automatic backups
linkbert apply --min-score 0.40

# 6. If needed, instant one-command rollback to original files
linkbert rollback

# 7. Discover external citation and authority link opportunities
linkbert external --targets references.txt
```

---

## 2. Core Commands Reference

### `linkbert scan <folder-or-url>`
Crawls a live site (via XML sitemaps or BFS crawl) or walks a local static build folder (`dist/`, `out/`, `build/`, `.output/`, etc.). Builds semantic vector embeddings and initial graph topology.

```bash
linkbert scan https://example.com --max-pages 300
linkbert scan ./dist --index .my_index
```

### `linkbert graph`
Calculates internal link equity and PageRank using Markov chain power iteration. Detects:
* PageRank distribution across all pages
* Inbound and outbound internal link counts
* Orphan pages (0 inbound links) and under-linked pages
* Topic silos / clusters

```bash
linkbert graph
linkbert graph --json
```

### `linkbert suggest`
Executes the Hybrid Discovery Engine combining direct mentions, calibrated LinkBERT, and semantic search:
* Mode `--mode hybrid` (default): Best of both worlds (direct mentions + neural LinkBERT).
* Mode `--mode mentions`: Direct target keyphrase matches.
* Mode `--mode linkbert`: Pure neural contextual anchors.
* Flag `--rerank`: Activates neural cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranking for joint context-to-target semantic validation.
* Automatically recalculates **Projected PageRank** to demonstrate SEO equity boost.
* Generates `report.html`, `suggestions.json`, `suggestions.csv`, and `graph_metrics.json`.

```bash
linkbert suggest --min-score 0.40 --per-target 2
linkbert suggest --rerank
linkbert suggest --page /blog/
```

### `linkbert clusters`
Detects topical silos and community structure across the site using either URL path hierarchy or graph-based Louvain community detection:
* Computes intra-silo cohesion scores (`internal / (internal + external)`)
* Identifies the primary hub page (highest PageRank node) for each cluster
* Detects silo violations where deep pages link cross-cluster without hub routing

```bash
linkbert clusters
linkbert clusters --method louvain
linkbert clusters --json
```

### `linkbert anchors`
Calculates Shannon entropy across anchor text variations for every target page to prevent Google Penguin algorithmic over-optimization penalties:
* Flags targets where anchor entropy < 1.5 bits
* Flags targets where exact title match ratio exceeds 60%
* Provides automated diversity recommendations (branded, partial-match, and topical variations)

```bash
linkbert anchors
linkbert anchors --json
```

### `linkbert cannibalization`
Parses Google Search Console Performance exports (Queries + Landing Pages) to identify queries where multiple pages compete for organic rankings:
* Quantifies split clicks and impressions
* Recommends consolidating internal link equity onto the highest-ranking canonical URL
* Directs internal anchor text from secondary pages to the primary ranking page

```bash
linkbert cannibalization --csv search_console_queries.csv
linkbert cannibalization --csv export.csv --min-impressions 20
```

### `linkbert health`
Performs automated link health validation on all recommended internal and external link targets:
* Verifies local files exist in Mode A
* Performs HTTP HEAD requests to ensure 200 OK responses in web mode
* Flags 4xx/5xx dead links and excessive redirect chains (> 2 hops)

```bash
linkbert health
```

### `linkbert protocol`
Outputs or saves the comprehensive 7-phase Agent Execution Protocol runbook for autonomous AI agents (Phase 0 Project Discovery through Phase 7 Verification):

```bash
linkbert protocol
linkbert protocol --save
```

### `linkbert apply`
Automated link placement engine. Modifies HTML and Markdown (`.md`, `.mdx`) files directly:
* Guaranteed to never link inside existing `<a>`, headers `<h1>`-`<h6>`, scripts, styles, code blocks, or inline code.
* Automatically creates `.bak` backups.
* Supports `--dry-run` and `--diff` for inspection.

```bash
linkbert apply --dry-run --diff
linkbert apply --min-score 0.45
linkbert apply --page /senior-pets/
```

### `linkbert rollback` & `linkbert commit`
* `linkbert rollback`: Instantly restores all `.bak` backup files to original condition.
* `linkbert commit`: Cleans up `.bak` backup files once link modifications are approved.

### `linkbert external`
Scans content for:
* Statistical claims and figures requiring external research citations for Google E-E-A-T.
* Recognized industry governing bodies, regulations, and standards.
* Curated external target matching from a user-supplied text file.

```bash
linkbert external --targets authority_urls.txt
```

### `linkbert gaps`
Exposes link equity gaps, highlighting orphan pages with zero internal links and recommending the highest-PageRank parent pages to link down to them.

```bash
linkbert gaps --top 20
```

### `linkbert doctor`
Verifies environment health, PyTorch CUDA acceleration, GPU memory, Transformers, and SentenceTransformers.

---

## 3. Operating Modes & BrowserOS Neo Integration

LinkGraph Pro supports two distinct operational modes designed for maximum flexibility across local codebases and live production websites:

### Mode A: Local Codebase
Use when you have the project repository on your local machine:
1. Direct the agent or CLI to the local codebase folder: `linkbert scan ./dist` or `linkbert scan ./my-site`.
2. LinkGraph Pro inspects HTML, Markdown, and MDX files, builds the graph, and discovers opportunities.
3. Apply links with AST safety and automatic `.bak` backups: `linkbert apply --dry-run --diff` then `linkbert apply`.
4. Run project build / tests to verify zero markup corruption.

### Mode B: URL Only (via BrowserOS Neo)
Use when you only have a live website URL (e.g. `https://example.com`):
1. **Rendered DOM Extraction**: If the site is JavaScript-heavy (React, Next.js, Vue), BrowserOS Neo renders the page, executes scripts, and extracts post-hydration visible text blocks and internal links.
2. **Google Search Console**: BrowserOS Neo leverages persistent user sessions to access Google Search Console Performance data (clicks, impressions, position) without needing API credentials.
3. **SERP Competitor Guard**: When recommending external citation targets, BrowserOS Neo queries Google/Bing for the site's primary ranking queries to ensure you never link to direct organic competitors.
4. **Link Health Checks**: Verifies all external and internal destinations return clean `200 OK` status codes with no redirect loops.

---

## 4. SEO Quality Rules Enforced

* **Anti-Cannibalization**: Suppresses generic broad anchor terms from linking down to specific sub-niche pages.
* **Anchor Text Diversity**: Prevents over-optimizing target pages with repetitive anchors via Shannon entropy guard.
* **Orphan Rescue Priority**: Strongly favors routing link equity down to isolated, unlinked pages.
* **No Heading Links**: Protects `<h1>`-`<h6>` tags from link pollution.
* **Cluster Silo Preservation**: Enhances intra-silo topical authority with hub-routed cross-silo governance.
