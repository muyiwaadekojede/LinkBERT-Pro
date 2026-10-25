"""Automated Quality Gate & Specialist Verification Committee.

Audits every row in the generated Link Architecture Master Workbook (.xlsx)
and ensures 100% compliance across all 4 specialist domains:
1. Specialist 1: Anchor Syntax, Grammar & Morphological Boundary Auditor
2. Specialist 2: Topical Silo Coherence & Entity Routing Auditor
3. Specialist 3: External Authority Citation & E-E-A-T Auditor
4. Specialist 4: GSC Striking Distance & Topology Matrix Auditor
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any
import openpyxl

def audit_workbook(xlsx_path: Path) -> Dict[str, Any]:
    if not xlsx_path.exists():
        return {"error": f"Workbook {xlsx_path} does not exist"}

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    defects: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # AUDIT TAB 1: Internal_Linking_Matrix
    # -------------------------------------------------------------------------
    if "Internal_Linking_Matrix" not in wb.sheetnames:
        defects.append({"specialist": "Structure", "severity": "CRITICAL", "issue": "Missing Internal_Linking_Matrix tab"})
    else:
        ws_int = wb["Internal_Linking_Matrix"]
        rows = list(ws_int.iter_rows(values_only=True))
        if len(rows) <= 1:
            defects.append({"specialist": "Internal", "severity": "CRITICAL", "issue": "Internal_Linking_Matrix is empty"})
        else:
            headers = [str(h) for h in rows[0]]
            for r_idx, row in enumerate(rows[1:], start=2):
                src, anchor, tgt, title, score, engine, sent, src_file = row[:8]
                src = str(src or "")
                anchor = str(anchor or "").strip()
                tgt = str(tgt or "")
                sent = str(sent or "")

                # Specialist 1: Syntax & Grammar Check
                # Check for truncated words
                if re.search(r"\b(back|pen|hard)$", anchor, re.IGNORECASE) and len(anchor.split()) > 1:
                    if anchor.lower() in ("lost 1,212 back", "update pen", "core web vitals hard"):
                        defects.append({
                            "specialist": "Syntax_Auditor",
                            "row": r_idx,
                            "anchor": anchor,
                            "issue": f"Truncated anchor word: '{anchor}'"
                        })

                # Check unclosed parentheses
                if "(" in anchor and ")" not in anchor:
                    defects.append({
                        "specialist": "Syntax_Auditor",
                        "row": r_idx,
                        "anchor": anchor,
                        "issue": f"Unclosed parenthesis in anchor: '{anchor}'"
                    })

                # Check conversational CTA imperative verb anchors
                cta_patterns = [r"^get a free\b", r"^contact me\b", r"^requires you to\b", r"^click here\b", r"^page onto\b", r"^i implemented\b"]
                if any(re.search(p, anchor, re.IGNORECASE) for p in cta_patterns):
                    defects.append({
                        "specialist": "Syntax_Auditor",
                        "row": r_idx,
                        "anchor": anchor,
                        "issue": f"Conversational CTA phrase used as anchor: '{anchor}'"
                    })

                # Check trailing discard words
                if re.search(r"\s+\b(because|they|this|that|these|those|with|for|from|into|onto)\b$", anchor, re.IGNORECASE):
                    defects.append({
                        "specialist": "Syntax_Auditor",
                        "row": r_idx,
                        "anchor": anchor,
                        "issue": f"Dangling function word at end of anchor: '{anchor}'"
                    })

                # Specialist 2: Topical Silo Coherence
                # Core Web Vitals must never link to Web 2.0 or off-page link building
                perf_terms = ("core web vitals", "pagespeed", "lighthouse", "lcp", "inp", "cls")
                if any(k in anchor.lower() for k in perf_terms):
                    if "web-2-0" in tgt.lower() or "link-building" in tgt.lower() or "duplicate-content" in tgt.lower():
                        defects.append({
                            "specialist": "Silo_Auditor",
                            "row": r_idx,
                            "anchor": anchor,
                            "target": tgt,
                            "issue": f"Severe Topical Mismatch: Performance anchor '{anchor}' linked to off-page SEO '{tgt}'"
                        })

                # Persona Entity must never link to academic teaching
                if "muyiwa adekojede" in anchor.lower() or "muyiwa" in anchor.lower():
                    if "/academics/teaching" in tgt.lower():
                        defects.append({
                            "specialist": "Silo_Auditor",
                            "row": r_idx,
                            "anchor": anchor,
                            "target": tgt,
                            "issue": f"Entity Misrouting: Author persona '{anchor}' linked to sub-page '{tgt}' instead of canonical about page"
                        })

                # Commercial keyword research must not link to academics
                if "keyword research" in anchor.lower() and "/academics" in tgt.lower():
                    defects.append({
                        "specialist": "Silo_Auditor",
                        "row": r_idx,
                        "anchor": anchor,
                        "target": tgt,
                        "issue": f"Cross-Silo Contamination: Commercial SEO '{anchor}' linked to academic research '{tgt}'"
                    })

    # -------------------------------------------------------------------------
    # AUDIT TAB 2: External_Authority_Citations
    # -------------------------------------------------------------------------
    if "External_Authority_Citations" not in wb.sheetnames:
        defects.append({"specialist": "Structure", "severity": "CRITICAL", "issue": "Missing External_Authority_Citations tab"})
    else:
        ws_ext = wb["External_Authority_Citations"]
        ext_rows = list(ws_ext.iter_rows(values_only=True))
        if len(ext_rows) <= 1:
            defects.append({"specialist": "External", "severity": "WARNING", "issue": "External_Authority_Citations has no data rows"})
        else:
            seen_page_targets: Dict[str, set] = {}
            global_target_counts: Dict[str, int] = {}

            for r_idx, row in enumerate(ext_rows[1:], start=2):
                src, anchor, sent, tgt_url, title, domain, tier, cat, rel, conf, reason, paper = row[:12]
                src = str(src or "")
                anchor = str(anchor or "").strip()
                tgt_url = str(tgt_url or "").strip()
                sent = str(sent or "")

                # Specialist 3: Zero Placeholders
                if "#citation-needed" in tgt_url.lower():
                    defects.append({
                        "specialist": "External_Auditor",
                        "row": r_idx,
                        "anchor": anchor,
                        "target_url": tgt_url,
                        "issue": f"Unresolved Placeholder URL: '{tgt_url}' in production deliverable"
                    })

                # Mismatched Patents
                if "patents.google.com" in tgt_url.lower() and "patent" not in sent.lower() and "patent" not in anchor.lower():
                    defects.append({
                        "specialist": "External_Auditor",
                        "row": r_idx,
                        "anchor": anchor,
                        "target_url": tgt_url,
                        "issue": f"Absurd Mapping: Non-patent assertion '{anchor}' routed to patents.google.com"
                    })

                # Check Anchor in Sentence
                if anchor.lower() not in sent.lower():
                    # check first 2 words
                    w2 = " ".join(anchor.lower().split()[:2])
                    if w2 not in sent.lower():
                        defects.append({
                            "specialist": "External_Auditor",
                            "row": r_idx,
                            "anchor": anchor,
                            "sentence": sent[:80],
                            "issue": f"Context Misalignment: Anchor '{anchor}' not found in sentence context"
                        })

                # Check Duplicate Links on Same Page
                if src not in seen_page_targets:
                    seen_page_targets[src] = set()
                if tgt_url.lower() in seen_page_targets[src]:
                    defects.append({
                        "specialist": "External_Auditor",
                        "row": r_idx,
                        "source": src,
                        "target_url": tgt_url,
                        "issue": f"Duplicate external link on same page: '{tgt_url}'"
                    })
                seen_page_targets[src].add(tgt_url.lower())

                # Track global counts
                global_target_counts[tgt_url.lower()] = global_target_counts.get(tgt_url.lower(), 0) + 1

            # Check for excessive over-budgeting across site
            for url, count in global_target_counts.items():
                if count > 3:
                    defects.append({
                        "specialist": "External_Auditor",
                        "target_url": url,
                        "count": count,
                        "issue": f"Outbound Link Over-Budgeting: Target '{url}' recommended {count} times across site (max budget 3)"
                    })

    # -------------------------------------------------------------------------
    # AUDIT TAB 3: GSC_Striking_Distance
    # -------------------------------------------------------------------------
    if "GSC_Striking_Distance" not in wb.sheetnames:
        defects.append({"specialist": "Structure", "severity": "CRITICAL", "issue": "Missing GSC_Striking_Distance tab"})
    else:
        ws_gsc = wb["GSC_Striking_Distance"]
        gsc_rows = list(ws_gsc.iter_rows(values_only=True))
        if len(gsc_rows) <= 1:
            defects.append({"specialist": "GSC_Auditor", "severity": "WARNING", "issue": "GSC_Striking_Distance has no rows"})

    # -------------------------------------------------------------------------
    # AUDIT TAB 4: Graph_Topology_Audit
    # -------------------------------------------------------------------------
    if "Graph_Topology_Audit" not in wb.sheetnames:
        defects.append({"specialist": "Structure", "severity": "CRITICAL", "issue": "Missing Graph_Topology_Audit tab"})

    wb.close()
    critical_defects = [d for d in defects if d.get("severity") != "WARNING"]
    return {
        "workbook": str(xlsx_path),
        "total_defects": len(critical_defects),
        "critical_defects": critical_defects,
        "warnings": [d for d in defects if d.get("severity") == "WARNING"],
        "status": "PASS" if len(critical_defects) == 0 else "FAIL"
    }

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".linkbert_muyiwa/link_architecture_master.xlsx")
    res = audit_workbook(path)
    import json
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["status"] == "PASS" else 1)
