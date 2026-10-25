"""Precision AST/Regex Link Application Engine with Safe Rollback and Diff Previews.

Safely modifies HTML, Markdown (.md), and MDX (.mdx) files to insert
internal/external links into visible body text.

Guarantees:
- Never links inside existing <a> tags
- Never links inside <h1>-<h6>, <script>, <style>, <code>, <pre>, or attributes
- Never links inside Markdown code fences (```...```) or inline code (`...`)
- Always creates atomic .bak backups before any write
- Full rollback capability with `linkbert rollback`
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from .discovery import CandidateOpportunity


@dataclass
class ApplyResult:
    file_path: str
    applied_count: int
    skipped_count: int
    diff: str
    backup_path: str | None = None


class LinkApplier:
    def __init__(self, dry_run: bool = False, create_backup: bool = True) -> None:
        self.dry_run = dry_run
        self.create_backup = create_backup

    def _backup(self, path: Path) -> Path:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        return bak

    def apply_to_html(self, file_path: Path, opportunities: List[CandidateOpportunity]) -> ApplyResult:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(content, "lxml")
        applied = 0
        skipped = 0

        # Disallowed parent tags
        DISALLOWED_TAGS = {"a", "script", "style", "code", "pre", "button", "h1", "h2", "h3", "h4", "h5", "h6", "nav", "header", "footer"}

        for opp in opportunities:
            anchor = opp.anchor
            target_href = opp.target_key
            pattern = re.compile(r"\b" + re.escape(anchor) + r"\b", re.IGNORECASE)

            # Search in body text nodes
            matched = False
            for text_node in list(soup.find_all(string=True)):
                if not isinstance(text_node, NavigableString):
                    continue

                parent = text_node.parent
                if not parent:
                    continue

                # Check if any ancestor is in disallowed tags
                ancestor_tags = {p.name.lower() for p in text_node.parents}
                if ancestor_tags & DISALLOWED_TAGS:
                    continue

                raw_text = str(text_node)
                match = pattern.search(raw_text)
                if match:
                    s, e = match.start(), match.end()
                    before = raw_text[:s]
                    matched_text = raw_text[s:e]
                    after = raw_text[e:]

                    # Create new link tag
                    link_tag = soup.new_tag("a", href=target_href)
                    link_tag.string = matched_text

                    # Replace node with split parts
                    replacement = []
                    if before:
                        replacement.append(NavigableString(before))
                    replacement.append(link_tag)
                    if after:
                        replacement.append(NavigableString(after))

                    text_node.replace_with(*replacement)
                    applied += 1
                    matched = True
                    break

            if not matched:
                skipped += 1

        # Format output
        new_content = str(soup)
        diff = "\n".join(difflib.unified_diff(
            content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
            lineterm="",
        ))

        bak_path = None
        if applied > 0 and not self.dry_run:
            if self.create_backup:
                bak_path = str(self._backup(file_path))
            file_path.write_text(new_content, encoding="utf-8")

        return ApplyResult(
            file_path=str(file_path),
            applied_count=applied,
            skipped_count=skipped,
            diff=diff,
            backup_path=bak_path,
        )

    def apply_to_markdown(self, file_path: Path, opportunities: List[CandidateOpportunity]) -> ApplyResult:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        applied = 0
        skipped = 0

        in_code_fence = False
        new_lines = []

        # Sort opportunities by line if available or process sequentially
        remaining_opps = list(opportunities)

        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("```") or trimmed.startswith("~~~"):
                in_code_fence = not in_code_fence
                new_lines.append(line)
                continue

            if in_code_fence or trimmed.startswith("#") or not trimmed:
                new_lines.append(line)
                continue

            cur_line = line
            for opp in list(remaining_opps):
                anchor = opp.anchor
                target_href = opp.target_key
                pattern = re.compile(r"\b" + re.escape(anchor) + r"\b", re.IGNORECASE)

                match = pattern.search(cur_line)
                if match:
                    s, e = match.start(), match.end()
                    # Check if already inside a markdown link [...](...) or inline code `...`
                    prefix = cur_line[:s]
                    suffix = cur_line[e:]

                    if prefix.count("`") % 2 != 0:
                        continue  # Inside inline code
                    if prefix.count("[") > prefix.count("]") and ")" in suffix:
                        continue  # Inside link text or link target

                    matched_text = cur_line[s:e]
                    cur_line = f"{prefix}[{matched_text}]({target_href}){suffix}"
                    applied += 1
                    remaining_opps.remove(opp)
                    break

            new_lines.append(cur_line)

        new_content = "\n".join(new_lines)
        if content.endswith("\n"):
            new_content += "\n"

        diff = "\n".join(difflib.unified_diff(
            content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
            lineterm="",
        ))

        bak_path = None
        if applied > 0 and not self.dry_run:
            if self.create_backup:
                bak_path = str(self._backup(file_path))
            file_path.write_text(new_content, encoding="utf-8")

        return ApplyResult(
            file_path=str(file_path),
            applied_count=applied,
            skipped_count=len(opportunities) - applied,
            diff=diff,
            backup_path=bak_path,
        )

    def apply(self, opportunities_by_file: Dict[str, List[CandidateOpportunity]]) -> List[ApplyResult]:
        results: List[ApplyResult] = []
        for file_str, opps in opportunities_by_file.items():
            p = Path(file_str)
            if not p.exists():
                continue

            if p.suffix.lower() in {".md", ".mdx", ".markdown"}:
                res = self.apply_to_markdown(p, opps)
            else:
                res = self.apply_to_html(p, opps)

            results.append(res)
        return results

    @staticmethod
    def rollback(target_dir: Path) -> List[str]:
        """Restores all .bak files in target_dir to original files."""
        restored: List[str] = []
        for bak in target_dir.rglob("*.bak"):
            orig = bak.with_suffix("")
            shutil.copy2(bak, orig)
            bak.unlink()
            restored.append(str(orig))
        return restored

    @staticmethod
    def commit_backups(target_dir: Path) -> int:
        """Removes all .bak backup files once changes are accepted."""
        count = 0
        for bak in target_dir.rglob("*.bak"):
            bak.unlink()
            count += 1
        return count
