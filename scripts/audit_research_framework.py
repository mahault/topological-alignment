"""Static integrity audit for the research framework.

This checks repository consistency. It does not validate philosophical premises,
empirical claims, or the truth of external sources.
"""

from __future__ import annotations

import json
import hashlib
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


def experiment_ledger_errors() -> list[str]:
    errors: list[str] = []
    path = ROOT / "benchmarks" / "experiment_pipeline_ledger.json"
    ledger = json.loads(path.read_text("utf-8"))
    required = {
        "id", "kind", "artifact", "disposition", "evidence_status",
        "permitted_claim", "blockers",
    }
    seen: set[str] = set()
    for index, entry in enumerate(ledger.get("entries", [])):
        missing = required - set(entry)
        if missing:
            errors.append(f"experiment ledger entry {index}: missing {sorted(missing)}")
            continue
        if entry["id"] in seen:
            errors.append(f"experiment ledger: duplicate id {entry['id']!r}")
        seen.add(entry["id"])
        artifact = ROOT / entry["artifact"]
        if not artifact.exists():
            errors.append(
                f"experiment ledger {entry['id']}: missing artifact {entry['artifact']!r}"
            )
        if not entry["disposition"]:
            errors.append(f"experiment ledger {entry['id']}: empty disposition")
        if not entry["permitted_claim"].strip():
            errors.append(f"experiment ledger {entry['id']}: empty permitted claim")
    if not seen:
        errors.append("experiment ledger has no entries")
    return errors


def v0_result_errors() -> list[str]:
    errors: list[str] = []
    result_path = ROOT / "benchmarks" / "v0_measurement_identification_results.json"
    result = json.loads(result_path.read_text("utf-8"))
    if result.get("epistemic_status") != "design recovery only; not human evidence":
        errors.append("V0 result ledger overstates its epistemic status")
    gates = result.get("gates", {})
    component_gates = {key: value for key, value in gates.items() if key != "all_gates_pass"}
    if not component_gates or gates.get("all_gates_pass") != all(component_gates.values()):
        errors.append("V0 result ledger gate summary is inconsistent")
    source = ROOT / "experiments" / "exp0_measurement_identification.py"
    normalized = source.read_text("utf-8").replace("\r\n", "\n").encode("utf-8")
    actual_hash = hashlib.sha256(normalized).hexdigest().upper()
    expected_hash = result.get("run", {}).get("source_sha256")
    if actual_hash != expected_hash:
        errors.append("V0 result ledger source hash does not match simulation source")
    return errors


def v1_result_errors() -> list[str]:
    errors: list[str] = []
    result_path = ROOT / "benchmarks" / "v1_multiscale_enabling_results.json"
    result = json.loads(result_path.read_text("utf-8"))
    if result.get("epistemic_status") != "finite model evidence only; no human or moral validation":
        errors.append("V1 result ledger overstates its epistemic status")
    gates = result.get("gates", {})
    component_gates = {key: value for key, value in gates.items() if key != "all_gates_pass"}
    if not component_gates or gates.get("all_gates_pass") != all(component_gates.values()):
        errors.append("V1 result ledger gate summary is inconsistent")
    source = ROOT / "experiments" / "exp_v1_multiscale_enabling.py"
    normalized = source.read_text("utf-8").replace("\r\n", "\n").encode("utf-8")
    actual_hash = hashlib.sha256(normalized).hexdigest().upper()
    expected_hash = result.get("run", {}).get("source_sha256")
    if actual_hash != expected_hash:
        errors.append("V1 result ledger source hash does not match simulation source")
    return errors


def v2_result_errors() -> list[str]:
    errors: list[str] = []
    result_path = ROOT / "benchmarks" / "v2_virtue_attractor_results.json"
    result = json.loads(result_path.read_text("utf-8"))
    expected_status = "constructed dynamical simulation; no human or moral validation"
    if result.get("epistemic_status") != expected_status:
        errors.append("V2 result ledger overstates its epistemic status")
    gates = result.get("gates", {})
    component_gates = {key: value for key, value in gates.items() if key != "all_gates_pass"}
    if not component_gates or gates.get("all_gates_pass") != all(component_gates.values()):
        errors.append("V2 result ledger gate summary is inconsistent")
    source = ROOT / "experiments" / "exp_v2_virtue_attractor.py"
    normalized = source.read_text("utf-8").replace("\r\n", "\n").encode("utf-8")
    actual_hash = hashlib.sha256(normalized).hexdigest().upper()
    expected_hash = result.get("run", {}).get("source_sha256")
    if actual_hash != expected_hash:
        errors.append("V2 result ledger source hash does not match simulation source")
    return errors


def v2_robustness_result_errors() -> list[str]:
    errors: list[str] = []
    result_path = ROOT / "benchmarks" / "v2_robustness_sweep_results.json"
    result = json.loads(result_path.read_text("utf-8"))
    expected_status = "constructed parameter sweep; no human or moral validation"
    if result.get("epistemic_status") != expected_status:
        errors.append("V2 robustness ledger overstates its epistemic status")
    gates = result.get("gates", {})
    component_gates = {key: value for key, value in gates.items() if key != "all_gates_pass"}
    if not component_gates or gates.get("all_gates_pass") != all(component_gates.values()):
        errors.append("V2 robustness ledger gate summary is inconsistent")
    run = result.get("run", {})
    for source_name, hash_name in (
        ("experiments/exp_v2_robustness_sweep.py", "source_sha256"),
        ("experiments/exp_v2_virtue_attractor.py", "model_source_sha256"),
    ):
        source = ROOT / source_name
        normalized = source.read_text("utf-8").replace("\r\n", "\n").encode("utf-8")
        actual_hash = hashlib.sha256(normalized).hexdigest().upper()
        if actual_hash != run.get(hash_name):
            errors.append(f"V2 robustness ledger hash does not match {source_name}")
    return errors


def research_guide_errors() -> list[str]:
    errors: list[str] = []
    notebook_path = ROOT / "notebooks" / "where_we_are.ipynb"
    html_path = ROOT / "notebooks" / "where_we_are.html"
    builder_path = ROOT / "notebooks" / "build_where_we_are.py"
    for path in (notebook_path, html_path, builder_path):
        if not path.exists():
            errors.append(f"research guide: missing {path.relative_to(ROOT)}")
    if errors:
        return errors
    notebook = json.loads(notebook_path.read_text("utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    expected_claims = {f"C{index:02d}" for index in range(1, 21)}
    present_claims = set(re.findall(r"^### (C\d{2}) —", source, flags=re.MULTILINE))
    if present_claims != expected_claims:
        errors.append(
            "research guide: canonical claim set differs from C01–C20 "
            f"(missing={sorted(expected_claims - present_claims)}, "
            f"extra={sorted(present_claims - expected_claims)})"
        )
    output_errors = [
        output
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if output_errors:
        errors.append("research guide: executed notebook contains error outputs")
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells or any(cell.get("execution_count") is None for cell in code_cells):
        errors.append("research guide: not every code cell has an executed output state")
    html = html_path.read_text("utf-8")
    if html.count("function Animation(frames") < 2:
        errors.append("research guide: fewer than two embedded animations")
    if 'name="static-math-renderer"' not in html:
        errors.append("research guide: equations are not statically rendered")
    if html.count('class="static-math ') < 30:
        errors.append("research guide: fewer than thirty embedded equation SVGs")
    if "cdnjs.cloudflare.com/ajax/libs/mathjax" in html:
        errors.append("research guide: still depends on remote MathJax")
    for heading in ("What counts as a proof here?", "Master claim register", "Experiment genealogy and setup"):
        if heading not in html:
            errors.append(f"research guide: HTML missing section {heading!r}")
    return errors


def run() -> list[str]:
    return [
        *markdown_link_errors(),
        *bibliography_errors(),
        *lean_errors(),
        *toolchain_errors(),
        *experiment_ledger_errors(),
        *v0_result_errors(),
        *v1_result_errors(),
        *v2_result_errors(),
        *v2_robustness_result_errors(),
        *research_guide_errors(),
    ]


if __name__ == "__main__":
    failures = run()
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASS: links, citations, Lean placeholders, toolchain, and result ledgers")
