#!/usr/bin/env python3
# Final evidence-faithful manuscript validator.
"""Validate the evidence-faithful GCTR reliability-audit manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
MAIN = MANUSCRIPT / "main.tex"
BIB = MANUSCRIPT / "refs.bib"
BBL = MANUSCRIPT / "main.bbl"
PDF = MANUSCRIPT / "main.pdf"
ARCHIVE = MANUSCRIPT / "arxiv-source-gctr.zip"
MANIFEST = MANUSCRIPT / "portfolio_artifacts_manifest.json"
EXPECTED_TITLE = (
    "A Reproducible Reliability Audit of "
    "Graph-Conditioned Optimizer Routing for QAOA"
)
EXPECTED_MAIN_SECTIONS = [
    "Results",
    "Discussion",
    "Methods",
    "Data availability",
    "Code availability",
    "Author contributions",
    "Competing interests",
]
BANNED_REFERENCE = "2607." + "06758"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def without_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def normalized_title(source: str) -> str:
    match = re.search(r"\\title\{(.*?)\}\s*\\author", source, flags=re.S)
    if not match:
        return ""
    title = match.group(1).replace(r"\textbf{", "")
    title = title.replace("\\\\", " ").replace("{", "").replace("}", "")
    return " ".join(title.split())


def word_count(source: str) -> int:
    source = re.sub(
        r"\\(?:cite|ref|eqref|label)\w*(?:\[[^\]]*\])?\{[^{}]*\}",
        " ",
        source,
    )
    source = re.sub(r"\\[A-Za-z@]+(?:\[[^\]]*\])?", " ", source)
    source = re.sub(r"[{}$\\_^~]", " ", source)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", source))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included_sources(source: str) -> dict[str, Path]:
    source = without_comments(source)
    relative = set(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source)
    )
    relative.update(re.findall(r"\\input\{([^{}]+)\}", source))
    paths: dict[str, Path] = {}
    for name in sorted(relative):
        relative_path = Path(name)
        require(not relative_path.is_absolute(), f"absolute include path: {name}")
        require(".." not in relative_path.parts, f"parent traversal in include: {name}")
        paths[name] = MANUSCRIPT / relative_path
    return paths


def structural_report(source: str) -> dict[str, object]:
    split = source.find(r"\appendix")
    require(split >= 0, "appendix transition is missing")
    main = source[:split]
    appendix = source[split:]
    sections = re.findall(r"^\\section\{([^{}]+)\}", main, flags=re.M)
    require(r"\section{Introduction}" not in main, "Introduction must be unheaded")
    require(
        sections == EXPECTED_MAIN_SECTIONS,
        f"main section order differs: {sections}",
    )

    figures_main = len(re.findall(r"\\begin\{figure\*?\}", main))
    tables_main = len(re.findall(r"\\begin\{table\*?\}", main))
    figures_appendix = len(re.findall(r"\\begin\{figure\*?\}", appendix))
    tables_appendix = len(re.findall(r"\\begin\{table\*?\}", appendix))
    main_displays = figures_main + tables_main
    require(1 <= main_displays <= 6, f"main display count is {main_displays}")
    require(source.count(r"\begin{theorem}") >= 1, "tie-safe theorem is missing")
    require(source.count(r"\begin{proof}") >= 1, "theorem proof is missing")
    require(r"\section{Extended Data}" in appendix, "Extended Data section is missing")
    require(
        appendix.count("Extended Data Fig.") == figures_appendix,
        "appendix figures are not all designated as Extended Data",
    )
    require(
        appendix.count("Extended Data Table") == tables_appendix,
        "appendix tables are not all designated as Extended Data",
    )

    lower = source.lower()
    require("acknowledgement" not in lower, "acknowledgements must be omitted")
    require("acknowledgment" not in lower, "acknowledgments must be omitted")
    require(
        not re.search(r"\\section\*?\{funding\}", lower),
        "funding section must be omitted",
    )
    require("impact statement" not in lower, "impact statement must be omitted")
    return {
        "main_sections": len(sections),
        "main_figures": figures_main,
        "main_tables": tables_main,
        "main_displays": main_displays,
        "supplementary_figures": figures_appendix,
        "supplementary_tables": tables_appendix,
        "theorems": source.count(r"\begin{theorem}"),
        "proofs": source.count(r"\begin{proof}"),
    }


def evidence_fidelity_report(source: str) -> dict[str, object]:
    normalized = " ".join(source.split())
    required_literals = {
        "development opportunity": "0.10710",
        "confirmatory opportunity": "0.07115",
        "structural-zero count": "84 of 160",
        "nontrivial coverage": r"\frac{57}{76}=0.75",
        "calibration ties": "24 of 48",
        "family-size bootstrap": "eight family-by-size strata",
        "global graph filter": "de-duplication is global",
        "forced ER edge": r"inserts edge $(0,1)$",
        "legacy zero-use finding": (
            "selected zero times, deployed zero times, and best on zero"
        ),
        "Codex disclosure": "OpenAI Codex",
        "AI responsibility": "No AI system generated experimental data",
    }
    for label, literal in required_literals.items():
        require(
            literal in normalized,
            f"missing evidence-fidelity statement: {label}",
        )
    require(r"\Delta_i^{\rm sel}" in source, "selected-effect notation is missing")
    require(r"\Gamma_i" in source, "deployed-effect notation is missing")
    require(
        "certifies a negative predicted" not in source,
        "gate is incorrectly described as a certificate",
    )
    require(BANNED_REFERENCE not in source, "forbidden arXiv citation is present")
    require(source.count("OpenAI Codex") == 1, "Codex disclosure must occur once")
    disclosure = re.search(
        r"\\subsection\{AI-assisted manuscript and code preparation\}"
        r"(.*?)(?=\\section|\\subsection|\\appendix)",
        source,
        flags=re.S,
    )
    require(disclosure is not None, "AI-assistance disclosure is missing")
    disclosure_words = word_count(disclosure.group(1)) if disclosure else 0
    require(
        20 <= disclosure_words <= 80,
        f"AI-assistance disclosure has {disclosure_words} words",
    )
    return {
        "required_fidelity_statements": len(required_literals) + 3,
        "ai_disclosure_words": disclosure_words,
    }


def citation_report(source: str, bibliography: str) -> dict[str, int]:
    bib_keys = re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", bibliography, flags=re.M)
    require(len(bib_keys) == len(set(bib_keys)), "duplicate bibliography keys")
    groups = re.findall(
        r"\\cite(?:p|t|alp|alt|author|year|yearpar)?"
        r"(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}",
        source,
    )
    group_sizes = [len([key for key in group.split(",") if key.strip()]) for group in groups]
    require(all(1 <= size <= 3 for size in group_sizes), "citation group exceeds three works")
    cited = {
        key.strip()
        for group in groups
        for key in group.split(",")
        if key.strip()
    }
    require(len(cited) >= 20, f"only {len(cited)} unique works are cited")
    require(not (cited - set(bib_keys)), "citation key is missing from bibliography")
    require(r"\nocite" not in source, "nocite is not permitted")
    require(BANNED_REFERENCE not in bibliography, "forbidden arXiv entry is present")
    return {
        "bibliography_entries": len(bib_keys),
        "citation_commands": len(groups),
        "citation_mentions": sum(group_sizes),
        "unique_cited_works": len(cited),
    }


def validate_manifest() -> dict[str, int]:
    require(MANIFEST.is_file(), "portfolio artifact manifest is missing")
    manifest = json.loads(MANIFEST.read_text())
    files = manifest.get("files", {})
    require(files, "portfolio artifact manifest contains no files")
    for relative, expected in files.items():
        path = ROOT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(sha256(path) == expected, f"manifest hash differs: {relative}")
    return {"manifest_files": len(files)}


def archive_sources(source: str) -> dict[str, Path]:
    sources = {
        "main.tex": MAIN,
        "main.bbl": BBL,
        "refs.bib": BIB,
    }
    sources.update(included_sources(source))
    return dict(sorted(sources.items()))


def validate_archive(source: str, required: bool) -> dict[str, object]:
    if not required:
        return {"archive_checked": False}
    require(ARCHIVE.is_file(), "submission archive is missing")
    sources = archive_sources(source)
    with zipfile.ZipFile(ARCHIVE) as handle:
        names = [name for name in handle.namelist() if not name.endswith("/")]
        require(names == sorted(sources), "submission archive member set/order differs")
        for name, path in sources.items():
            require(path.is_file(), f"archive input is missing: {name}")
            require(handle.read(name) == path.read_bytes(), f"stale archive member: {name}")
    return {
        "archive_checked": True,
        "archive_members": len(sources),
        "archive_sha256": sha256(ARCHIVE),
    }


def validate(*, require_archive: bool = True) -> dict[str, object]:
    tex = without_comments(MAIN.read_text())
    bib = BIB.read_text()
    require(
        r"\documentclass[10pt,twocolumn]{article}" in tex,
        "expected two-column submission format is missing",
    )
    require(normalized_title(tex) == EXPECTED_TITLE, "visible title differs")
    require(f"pdftitle={{{EXPECTED_TITLE}}}" in tex, "PDF metadata title differs")

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    require(abstract is not None, "abstract is missing")
    abstract_words = word_count(abstract.group(1)) if abstract else 0
    require(100 <= abstract_words <= 150, f"abstract has {abstract_words} words")

    included = included_sources(tex)
    for name, path in included.items():
        require(path.is_file(), f"included source is missing: {name}")
    labels = re.findall(r"\\label\{([^{}]+)\}", tex)
    require(len(labels) == len(set(labels)), "duplicate LaTeX labels")
    refs = re.findall(r"\\(?:eq)?ref\{([^{}]+)\}", tex)
    require(not (set(refs) - set(labels)), "reference points to a missing label")
    require(PDF.is_file() and PDF.stat().st_size > 100_000, "compiled PDF is missing")
    require(PDF.stat().st_mtime >= MAIN.stat().st_mtime, "compiled PDF is older than source")

    report = {
        "abstract_words": abstract_words,
        "included_sources": len(included),
        **structural_report(tex),
        **evidence_fidelity_report(tex),
        **citation_report(tex, bib),
        **validate_manifest(),
        **validate_archive(tex, require_archive),
    }
    if BBL.is_file():
        bibitems = len(re.findall(r"^\\bibitem", BBL.read_text(), flags=re.M))
        require(
            bibitems == report["unique_cited_works"],
            "compiled bibliography is stale",
        )
        report["compiled_bibitems"] = bibitems
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        validate(require_archive=not args.skip_archive),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
