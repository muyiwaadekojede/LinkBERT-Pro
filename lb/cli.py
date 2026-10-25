"""LinkGraph Pro CLI — Full-Spectrum SEO Link Architecture, Graph Intelligence & Automated Optimization.

Powered by:
- Calibrated LinkBERT Neural Token Classifier (adaptive paragraph-level quantile scoring)
- Hybrid Tri-Engine Discovery (Direct Mention + LinkBERT + Semantic Conceptual)
- Internal PageRank & Link Equity Simulation (Markov chain power iteration)
- Topic Cluster Detection & Silo Enforcement (URL hierarchy + Louvain community detection)
- Anchor Diversity & Shannon Entropy Guard (anti-Penguin over-optimization detection)
- AST-Safe Application & Rollback Engine (HTML + Markdown + MDX with atomic .bak backups)
- External Citation & E-E-A-T Authority Matcher (entity grounding + SERP competitor guard)
- BrowserOS Neo Integration (rendered DOM, Google Search Console, live SERP verification)
- Keyword Cannibalization Detection (Search Console query-to-URL conflict matrix)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from . import __version__
from .apply import LinkApplier
from .discovery import CandidateOpportunity, HybridDiscoveryEngine
from .external import ExternalLinkEngine
from .extract import parse_file
from .graph import LinkGraph
from .model import LinkPredictor
from .report import write_all
from .site import Page, SiteIndex, build

DEFAULT_INDEX = ".linkbert"


def _index_path(arg: str | None) -> Path:
    return Path(arg or os.environ.get("LINKBERT_INDEX", DEFAULT_INDEX)).expanduser().resolve()


def _load(arg: str | None) -> SiteIndex:
    p = _index_path(arg)
    if not (p / "site.json").exists():
        sys.exit(f"No index at {p}. Run: linkbert scan <folder-or-url>")
    return SiteIndex.load(p)


def _predictor(args, load_weights: bool = True) -> LinkPredictor | None:
    if not load_weights:
        return None
    try:
        return LinkPredictor(device=args.device, max_length=args.max_length)
    except Exception as e:
        print(f"Warning: Could not load LinkBERT weights ({e}). Running in target mention mode.", file=sys.stderr)
        return None


# ---------------------------------------------------------------- commands
def cmd_scan(args) -> None:
    out = _index_path(args.index)
    t0 = time.time()
    print(f"[*] Scanning {args.target} ...")
    idx = build(args.target, out, max_pages=args.max_pages, max_depth=args.depth, embed=not args.no_embed)
    dt = time.time() - t0
    print(f"[+] Indexed {len(idx)} pages in {dt:.1f}s -> {out}")
    if idx.pages:
        print(f"    Sample: {idx.pages[0].key}  ({idx.pages[0].title[:60]})")

    # Run initial graph analysis
    graph = LinkGraph(idx)
    metrics = graph.analyze()
    print(f"[+] Graph Topology: {len(metrics.orphan_pages)} orphan pages detected, {len(metrics.clusters)} clusters.")


def cmd_analyze(args) -> None:
    src = args.source
    if src == "-":
        text = sys.stdin.read()
        title = "stdin"
    else:
        p = Path(src)
        if p.is_file():
            doc = parse_file(p)
            text = doc.text
            title = doc.title or p.name
        else:
            text = src
            title = "inline text"
    if not text.strip():
        sys.exit("Nothing to analyze.")

    pred = _predictor(args, load_weights=True)
    if pred is None:
        sys.exit("LinkBERT model unavailable.")

    spans = pred.predict(text, high=args.high, low=args.low, min_words=args.min_words, max_words=args.max_words)
    if args.json:
        print(json.dumps([s.as_dict() for s in spans], indent=2, ensure_ascii=False))
        return

    print(f"{title} · {len(text.split())} words · {len(spans)} anchor candidates\n")
    for s in sorted(spans, key=lambda x: -x.score):
        print(f"  {s.score:.2f}  {s.text!r}")
        if args.context:
            print(f"        …{s.sentence}")
    if args.annotate:
        print("\n" + pred.annotate(text, high=args.high, low=args.low, min_words=args.min_words, max_words=args.max_words))


def cmd_suggest(args) -> None:
    idx = _load(args.index)
    load_bert = (args.mode in {"hybrid", "linkbert"})
    pred = _predictor(args, load_weights=load_bert) if load_bert else None

    engine = HybridDiscoveryEngine(idx, predictor=pred)

    pages = idx.pages
    if args.page:
        pages = [p for p in pages if args.page.lower() in p.key.lower()]
        if not pages:
            sys.exit(f"No page matching {args.page!r}")

    if args.limit:
        pages = pages[: args.limit]

    all_opps: List[CandidateOpportunity] = []
    t0 = time.time()

    for i, page in enumerate(pages, 1):
        if args.mode == "mentions":
            opps = engine.find_direct_mentions(page, set())
        elif args.mode == "linkbert":
            opps = engine.find_linkbert_opportunities(page, set(), adaptive_threshold=args.high)
        else:
            opps = engine.discover_all(
                page,
                min_score=args.min_score,
                per_target_limit=args.per_target,
                skip_existing=getattr(args, "skip_existing", False),
            )

        all_opps.extend(opps)
        print(f"  [{i}/{len(pages)}] {page.key} -> {len(opps)} opportunities found", file=sys.stderr)

    all_opps.sort(key=lambda s: -s.score)
    if args.top:
        all_opps = all_opps[: args.top]

    if getattr(args, "rerank", False) and all_opps:
        print("[*] Re-ranking candidates with neural cross-encoder...", file=sys.stderr)
        all_opps = engine.rerank_with_cross_encoder(all_opps)

    # Calculate LinkGraph metrics with projected PageRank
    proposed_pairs = [(o.source_key, o.target_key) for o in all_opps]
    graph = LinkGraph(idx)
    metrics = graph.analyze(additional_links=proposed_pairs)

    # Load GSC queries if provided or auto-discovered
    gsc_queries = []
    gsc_file = getattr(args, "gsc", None)
    if not gsc_file:
        candidate_gsc = _index_path(args.index) / "gsc_performance.csv"
        if candidate_gsc.exists():
            gsc_file = str(candidate_gsc)
        elif Path("tests/muyiwa_gsc_performance.csv").exists() and "muyiwa" in str(_index_path(args.index)):
            gsc_file = "tests/muyiwa_gsc_performance.csv"
    if gsc_file and Path(gsc_file).exists():
        from .browser import parse_search_console_csv
        sc_rows = parse_search_console_csv(Path(gsc_file).read_text(encoding="utf-8", errors="ignore"))
        gsc_queries = [{
            "query": r.query,
            "page": r.page,
            "clicks": r.clicks,
            "impressions": r.impressions,
            "ctr": f"{r.ctr*100:.1f}%" if isinstance(r.ctr, float) and r.ctr < 1 else str(r.ctr),
            "position": r.position,
        } for r in sc_rows]

    stats = write_all(_index_path(args.index), idx, all_opps, metrics=metrics, gsc_queries=gsc_queries)
    dt = time.time() - t0

    print(f"\n[+] {stats['opportunities']} internal link opportunities generated across {len(pages)} pages in {dt:.1f}s")
    print(f"[+] {stats.get('external_opportunities', 0)} external authority citations discovered")
    print(f"[+] Interactive Visual Dashboard: {stats['report']}")
    print(f"[+] Multi-Tab Excel Master Workbook: {stats.get('excel')}")
    print(f"[+] Internal Links CSV: {stats.get('internal_csv')}")
    print(f"[+] External Citations CSV: {stats.get('external_csv')}")
    print(f"[+] Full JSON Dataset: {stats['json']}")

    if args.json:
        print(json.dumps([o.as_dict() for o in all_opps], indent=2, ensure_ascii=False))


def cmd_apply(args) -> None:
    idx = _load(args.index)
    index_dir = _index_path(args.index)
    sug_file = index_dir / "suggestions.json"

    if not sug_file.exists():
        sys.exit(f"No suggestions found at {sug_file}. Run 'linkbert suggest' first.")

    data = json.loads(sug_file.read_text(encoding="utf-8"))
    raw_opps = data.get("opportunities") or data.get("suggestions") or []

    opps_by_file: Dict[str, List[CandidateOpportunity]] = {}
    for item in raw_opps:
        score = item.get("score", 0.0)
        if score < args.min_score:
            continue

        src_key = item.get("source") or item.get("source_key")
        if args.page and args.page.lower() not in src_key.lower():
            continue

        src_file = item.get("source_file")
        if not src_file:
            # Match to index page
            page = idx.by_key.get(src_key)
            if page:
                src_file = page.source

        if src_file:
            opp = CandidateOpportunity(
                source_key=src_key,
                source_file=src_file,
                anchor=item.get("anchor", ""),
                sentence=item.get("sentence", ""),
                target_key=item.get("target") or item.get("target_key", ""),
                target_title=item.get("target_title", ""),
                target_url=item.get("target_url", ""),
                score=score,
                engine=item.get("engine", "direct_mention"),
                block_index=item.get("block", 0),
            )
            opps_by_file.setdefault(src_file, []).append(opp)

    if not opps_by_file:
        print(f"No opportunities passed filter (min_score >= {args.min_score}).")
        return

    applier = LinkApplier(dry_run=args.dry_run, create_backup=not args.no_backup)
    results = applier.apply(opps_by_file)

    total_applied = sum(r.applied_count for r in results)
    mode_str = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode_str}Link Application Summary:")
    print(f"  Files targeted: {len(results)}")
    print(f"  Links successfully placed: {total_applied}")

    for r in results:
        if r.applied_count > 0:
            status = "would apply" if args.dry_run else "applied"
            print(f"  [+] {r.file_path}: {r.applied_count} links {status}")
            if args.diff and r.diff:
                print(r.diff)


def cmd_rollback(args) -> None:
    idx = _load(args.index)
    site_root = Path(idx.root) if idx.root and Path(idx.root).exists() else Path.cwd()
    restored = LinkApplier.rollback(site_root)
    if restored:
        print(f"[+] Restored {len(restored)} files from .bak backups:")
        for f in restored:
            print(f"    Restored: {f}")
    else:
        print("[-] No .bak backup files found to restore.")


def cmd_commit(args) -> None:
    idx = _load(args.index)
    site_root = Path(idx.root) if idx.root and Path(idx.root).exists() else Path.cwd()
    count = LinkApplier.commit_backups(site_root)
    print(f"[+] Committed changes. Removed {count} .bak backup files.")


def cmd_graph(args) -> None:
    idx = _load(args.index)
    graph = LinkGraph(idx)
    metrics = graph.analyze()

    if args.json:
        print(json.dumps({
            "page_rank": metrics.page_rank,
            "in_degrees": metrics.in_degrees,
            "out_degrees": metrics.out_degrees,
            "orphan_pages": metrics.orphan_pages,
            "under_linked_pages": metrics.under_linked_pages,
            "clusters": metrics.clusters,
        }, indent=2))
        return

    print("\n--- Site Internal PageRank Rankings (Top 15) ---")
    sorted_pr = sorted(metrics.page_rank.items(), key=lambda kv: -kv[1])
    for key, pr in sorted_pr[:15]:
        in_d = metrics.in_degrees.get(key, 0)
        out_d = metrics.out_degrees.get(key, 0)
        print(f"  {pr:>6.3f} PR | In: {in_d:>2} | Out: {out_d:>2} | {key}")

    print(f"\n--- Orphan Pages ({len(metrics.orphan_pages)}) ---")
    for key in metrics.orphan_pages:
        print(f"  [ORPHAN] 0 in-links: {key}")

    print(f"\n--- Topic Clusters ({len(metrics.clusters)}) ---")
    for cluster, pages in metrics.clusters.items():
        print(f"  Cluster '{cluster}': {len(pages)} pages")


def cmd_external(args) -> None:
    idx = _load(args.index)
    index_dir = _index_path(args.index)
    targets_file = Path(args.targets) if args.targets else None
    ext_engine = ExternalLinkEngine(targets_file)

    all_opps = []
    for page in idx.pages:
        opps = ext_engine.analyze_page(page)
        all_opps.extend(opps)

    out_csv = index_dir / "external_links.csv"
    ext_engine.export_csv(all_opps, out_csv)

    if args.json:
        print(json.dumps([o.as_dict() for o in all_opps], indent=2, ensure_ascii=False))
        return

    print(f"\n[+] Discovered {len(all_opps)} External Citation & Authority Link Opportunities:\n")
    for o in all_opps:
        tier_label = f"Tier {o.trust_tier}"
        print(f"  [{tier_label}] [{o.category.upper()}] ({o.confidence:.2f}) on {o.source_key}")
        print(f"    Anchor: {o.anchor!r} -> {o.target_url}")
        print(f"    Rel Attribute: rel=\"{o.recommended_rel}\"")
        print(f"    E-E-A-T Rationale: {o.reason}")
        print(f"    Academic / Patent Basis: {o.academic_basis}")
        print()

    print(f"[+] Saved structured external citations to: {out_csv}")


def cmd_gaps(args) -> None:
    idx = _load(args.index)
    graph = LinkGraph(idx)
    metrics = graph.analyze()

    print(f"\n{'Inbound':>8} {'PR':>6}  {'Page':<50} Title")
    sorted_pages = sorted(idx.pages, key=lambda p: (metrics.in_degrees.get(p.key, 0), -p.words))
    for p in sorted_pages[: args.top]:
        in_d = metrics.in_degrees.get(p.key, 0)
        pr = metrics.page_rank.get(p.key, 1.0)
        flag = "[ORPHAN]" if in_d == 0 and p.key != "/" else "        "
        print(f"{flag} {in_d:>3} {pr:>6.2f}  {p.key[:50]:<50} {p.title[:40]}")


def cmd_doctor(args) -> None:
    import sentence_transformers
    import torch
    import transformers

    print(f"LinkGraph Pro v{__version__}")
    print(f"Python      {sys.version.split()[0]}")
    print(f"PyTorch     {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU Device  {torch.cuda.get_device_name(0)}")
    print(f"Transformers {transformers.__version__}")
    print(f"SentenceTransformers {sentence_transformers.__version__}")
    print(f"Index Path  {_index_path(args.index)}")

    # Check for new modules
    modules = ["lb.browser", "lb.intelligence", "lb.orchestrator"]
    for mod in modules:
        try:
            __import__(mod)
            print(f"Module {mod:25s} [OK]")
        except ImportError as e:
            print(f"Module {mod:25s} [FAIL] ({e})")


def cmd_clusters(args) -> None:
    """Detect and display topic clusters (silos) in the site."""
    idx = _load(args.index)
    from .intelligence import detect_topic_clusters

    clusters = detect_topic_clusters(idx, method=args.method)

    if args.json:
        print(json.dumps([{
            "cluster_id": c.cluster_id,
            "name": c.name,
            "pages": c.pages,
            "hub_page": c.hub_page,
            "cohesion_score": c.cohesion_score,
            "internal_edges": c.internal_edge_count,
            "external_edges": c.external_edge_count,
        } for c in clusters], indent=2))
        return

    print(f"\n--- Topic Clusters ({len(clusters)} detected) ---\n")
    for c in clusters:
        cohesion_label = "STRONG" if c.cohesion_score > 0.6 else ("MODERATE" if c.cohesion_score > 0.3 else "WEAK")
        print(f"  Cluster {c.cluster_id}: {c.name}")
        print(f"    Pages: {len(c.pages)} | Hub: {c.hub_page}")
        print(f"    Cohesion: {c.cohesion_score:.2f} ({cohesion_label})")
        print(f"    Internal edges: {c.internal_edge_count} | External edges: {c.external_edge_count}")
        for pk in c.pages:
            marker = " [HUB]" if pk == c.hub_page else ""
            print(f"      * {pk}{marker}")
        print()


def cmd_anchors(args) -> None:
    """Analyze anchor text diversity and detect over-optimization."""
    idx = _load(args.index)
    index_dir = _index_path(args.index)
    sug_file = index_dir / "suggestions.json"

    if not sug_file.exists():
        sys.exit(f"No suggestions found at {sug_file}. Run 'linkbert suggest' first.")

    data = json.loads(sug_file.read_text(encoding="utf-8"))
    raw_opps = data.get("opportunities") or data.get("suggestions") or []

    proposed = [(item.get("source", ""), item.get("target", ""), item.get("anchor", "")) for item in raw_opps]

    from .intelligence import analyze_anchor_diversity
    profiles = analyze_anchor_diversity(idx, proposed)

    if args.json:
        print(json.dumps([{
            "target": p.target_key,
            "target_title": p.target_title,
            "anchors": p.anchors,
            "entropy": p.entropy,
            "exact_match_ratio": p.exact_match_ratio,
            "is_over_optimized": p.is_over_optimized,
            "recommendation": p.recommendation,
        } for p in profiles], indent=2))
        return

    print(f"\n--- Anchor Diversity Analysis ({len(profiles)} targets) ---\n")
    for p in profiles:
        status = "[OVER-OPTIMIZED]" if p.is_over_optimized else "[HEALTHY]"
        print(f"  {status}  {p.target_key}")
        print(f"    Title: {p.target_title[:60]}")
        print(f"    Entropy: {p.entropy:.3f} bits | Exact-match ratio: {p.exact_match_ratio:.0%}")
        print(f"    Anchors: {p.anchors}")
        print(f"    -> {p.recommendation}")
        print()


def cmd_protocol(args) -> None:
    """Display or save the agent execution protocol."""
    from .orchestrator import get_protocol, save_protocol

    if args.save:
        out = _index_path(args.index)
        path = save_protocol(out)
        print(f"Protocol saved to {path}")
    else:
        print(get_protocol())


def cmd_health(args) -> None:
    """Verify HTTP health of all recommended link targets."""
    idx = _load(args.index)
    index_dir = _index_path(args.index)
    sug_file = index_dir / "suggestions.json"

    if not sug_file.exists():
        sys.exit(f"No suggestions found. Run 'linkbert suggest' first.")

    data = json.loads(sug_file.read_text(encoding="utf-8"))
    raw_opps = data.get("opportunities") or data.get("suggestions") or []

    # Collect unique internal target paths
    internal_targets = set()
    for item in raw_opps:
        target = item.get("target", "")
        if target and not target.startswith(("http://", "https://")):
            internal_targets.add(target)

    print(f"\n--- Link Health Check ---")
    print(f"  Internal targets to verify: {len(internal_targets)}")

    # For local mode, verify files exist
    if idx.kind == "local":
        missing = []
        for target in internal_targets:
            page = idx.by_key.get(target)
            if not page:
                missing.append(target)
        found = len(internal_targets) - len(missing)
        print(f"  Internal: {found}/{len(internal_targets)} targets found in index")
        if missing:
            print(f"  [!] Missing targets:")
            for m in missing:
                print(f"      {m}")
    else:
        # For web mode, use HTTP verification
        from .browser import verify_link_health
        urls = [f"{idx.root.rstrip('/')}{t}" for t in internal_targets if not t.startswith("http")]
        results = verify_link_health(urls[:50])
        healthy = sum(1 for r in results if r["is_healthy"])
        print(f"  HTTP verification: {healthy}/{len(results)} healthy")
        for r in results:
            if not r["is_healthy"]:
                print(f"    [X] {r['url']} -> {r.get('status_code', 'unreachable')}")

    print(f"\n[+] Health check complete.")


def cmd_cannibalization(args) -> None:
    """Analyze Search Console Performance export for keyword cannibalization."""
    from .browser import parse_search_console_csv
    from .intelligence import detect_cannibalization

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV file not found: {csv_path}")

    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    rows = parse_search_console_csv(text)
    print(f"[*] Parsed {len(rows)} Search Console rows from {csv_path.name}")

    issues = detect_cannibalization(rows, min_impressions=args.min_impressions)

    if args.json:
        print(json.dumps([{
            "query": i.query,
            "severity": i.severity,
            "competing_pages": i.competing_pages,
            "recommendation": i.recommendation,
        } for i in issues], indent=2))
        return

    print(f"\n--- Keyword Cannibalization Report ({len(issues)} queries affected) ---\n")
    if not issues:
        print("  [OK] No keyword cannibalization detected above impression threshold.")
        return

    for issue in issues:
        marker = "[CRITICAL]" if issue.severity == "critical" else (
            "[WARNING]" if issue.severity == "warning" else "[INFO]"
        )
        print(f"  {marker} Query: '{issue.query}' ({len(issue.competing_pages)} competing pages)")
        for page_data in issue.competing_pages:
            pos = page_data.get("position", "?")
            clicks = page_data.get("clicks", 0)
            imps = page_data.get("impressions", 0)
            print(f"      pos {pos:<4} | {clicks:>3} clicks | {imps:>5} imps | {page_data.get('page')}")
        print(f"      -> {issue.recommendation}\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="linkbert", description="LinkGraph Pro — Full-Spectrum SEO Link Architecture & Optimization.")
    ap.add_argument("--version", action="version", version=f"LinkGraph Pro {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--index", help="Index folder (default .linkbert)")
        p.add_argument("--device", help="cpu or cuda")
        p.add_argument("--max-length", type=int, default=512)
        p.add_argument("--high", type=float, default=0.30)
        p.add_argument("--low", type=float, default=0.18)
        p.add_argument("--min-words", type=int, default=1)
        p.add_argument("--max-words", type=int, default=8)

    # scan
    p = sub.add_parser("scan", help="Index a local folder or live website")
    p.add_argument("target")
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--index")
    p.set_defaults(func=cmd_scan)

    # analyze
    p = sub.add_parser("analyze", help="Predict anchor spans for text or a file")
    p.add_argument("source", help="File path, '-' for stdin, or raw text")
    p.add_argument("--json", action="store_true")
    p.add_argument("--annotate", action="store_true")
    p.add_argument("--context", action="store_true")
    common(p)
    p.set_defaults(func=cmd_analyze)

    # suggest
    p = sub.add_parser("suggest", help="Generate optimized link opportunities")
    p.add_argument("--mode", choices=["hybrid", "mentions", "linkbert"], default="hybrid")
    p.add_argument("--page", help="Filter by source page key substring")
    p.add_argument("--min-score", type=float, default=0.28)
    p.add_argument("--per-target", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true", help="Skip target pages if already linked in body text")
    p.add_argument("--limit", type=int, help="Max pages to scan")
    p.add_argument("--top", type=int, help="Max total opportunities to keep")
    p.add_argument("--rerank", action="store_true", help="Re-rank opportunities using neural cross-encoder")
    p.add_argument("--gsc", help="Path to Google Search Console performance CSV")
    p.add_argument("--json", action="store_true")
    common(p)
    p.set_defaults(func=cmd_suggest)

    # apply
    p = sub.add_parser("apply", help="Safely apply links into source files with backups")
    p.add_argument("--dry-run", action="store_true", help="Preview diffs without modifying files")
    p.add_argument("--diff", action="store_true", help="Display full unified diffs")
    p.add_argument("--min-score", type=float, default=0.40)
    p.add_argument("--page", help="Only apply to source files matching page key")
    p.add_argument("--no-backup", action="store_true", help="Do not create .bak files")
    p.add_argument("--index")
    p.set_defaults(func=cmd_apply)

    # rollback
    p = sub.add_parser("rollback", help="Restore all files from .bak backups")
    p.add_argument("--index")
    p.set_defaults(func=cmd_rollback)

    # commit
    p = sub.add_parser("commit", help="Commit link changes and remove .bak files")
    p.add_argument("--index")
    p.set_defaults(func=cmd_commit)

    # graph
    p = sub.add_parser("graph", help="Display PageRank rankings and link equity flow")
    p.add_argument("--json", action="store_true")
    p.add_argument("--index")
    p.set_defaults(func=cmd_graph)

    # clusters
    p = sub.add_parser("clusters", help="Detect topic clusters (silos) and cohesion metrics")
    p.add_argument("--method", choices=["url_hierarchy", "louvain", "label_propagation"], default="url_hierarchy")
    p.add_argument("--json", action="store_true")
    p.add_argument("--index")
    p.set_defaults(func=cmd_clusters)

    # anchors
    p = sub.add_parser("anchors", help="Analyze anchor text diversity and over-optimization risk")
    p.add_argument("--json", action="store_true")
    p.add_argument("--index")
    p.set_defaults(func=cmd_anchors)

    # external
    p = sub.add_parser("external", help="Discover external citation and authority link opportunities")
    p.add_argument("--targets", help="Path to text file containing curated external URLs")
    p.add_argument("--json", action="store_true")
    p.add_argument("--index")
    p.set_defaults(func=cmd_external)

    # gaps
    p = sub.add_parser("gaps", help="List orphan and under-linked pages")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--json", action="store_true")
    p.add_argument("--index")
    p.set_defaults(func=cmd_gaps)

    # health
    p = sub.add_parser("health", help="Verify HTTP health of all recommended link targets")
    p.add_argument("--index")
    p.set_defaults(func=cmd_health)

    # protocol
    p = sub.add_parser("protocol", help="Display or save the agent execution protocol")
    p.add_argument("--save", action="store_true", help="Save protocol to index directory")
    p.add_argument("--index")
    p.set_defaults(func=cmd_protocol)

    # doctor
    p = sub.add_parser("doctor", help="Verify environment, modules, and CUDA capabilities")
    p.add_argument("--index")
    p.set_defaults(func=cmd_doctor)

    # cannibalization
    p = sub.add_parser("cannibalization", help="Detect keyword cannibalization from Search Console CSV")
    p.add_argument("--csv", required=True, help="Path to Search Console Performance CSV export")
    p.add_argument("--min-impressions", type=int, default=10, help="Minimum impressions threshold")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cannibalization)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
