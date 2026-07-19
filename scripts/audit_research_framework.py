"""Static integrity audit for the research framework.

This checks repository consistency. It does not validate philosophical premises,
empirical claims, or the truth of external sources.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown_link_errors() -> list[str]:
    errors: list[str] = []
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        text = document.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            target = target.strip().split()[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local_part = target.split("#", 1)[0]
            if not local_part:
                continue
            resolved = (document.parent / local_part).resolve()
            if not resolved.exists():
                errors.append(
                    f"{document.relative_to(ROOT)}: broken local link {target!r}"
                )
    return errors


def bibliography_errors() -> list[str]:
    manuscript = (ROOT / "tex" / "main.tex").read_text(encoding="utf-8")
    bibliography = (ROOT / "tex" / "references.bib").read_text(encoding="utf-8")
    cited: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", manuscript):
        cited.update(key.strip() for key in group.split(","))
    entries = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    return [f"tex/main.tex: missing BibTeX entry {key!r}" for key in sorted(cited - entries)]


def lean_errors() -> list[str]:
    errors: list[str] = []
    forbidden = re.compile(r"\b(sorry|admit|axiom)\b")
    for source in (ROOT / "formal").rglob("*.lean"):
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if forbidden.search(line):
                errors.append(
                    f"{source.relative_to(ROOT)}:{line_number}: placeholder or axiom"
                )
    return errors


def toolchain_errors() -> list[str]:
    errors: list[str] = []
    manifest = json.loads((ROOT / "formal" / "lake-manifest.json").read_text("utf-8"))
    formal_readme = (ROOT / "formal" / "README.md").read_text("utf-8").lower()
    proof_policy = (ROOT / "docs" / "MATHEMATICAL_PROOF_VALIDATION.md").read_text(
        "utf-8"
    ).lower()
    packages = manifest.get("packages", [])
    if packages and "mathlib was\nnot installed" in formal_readme:
        errors.append("formal README says Mathlib absent but manifest has packages")
    if not packages and "mathlib in the\nlake configuration" in proof_policy:
        errors.append("proof policy says Mathlib pinned but manifest has no packages")
    return errors


def run() -> list[str]:
    return [
        *markdown_link_errors(),
        *bibliography_errors(),
        *lean_errors(),
        *toolchain_errors(),
    ]


if __name__ == "__main__":
    failures = run()
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASS: links, citations, Lean placeholders, and toolchain status")
