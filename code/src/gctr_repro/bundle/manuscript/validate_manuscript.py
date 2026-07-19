#!/usr/bin/env python3
"""Validate the locked GCTR manuscript and its deterministic arXiv archive."""

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
POPULAR = MANUSCRIPT / "prx-popular-summary.tex"
PDF = MANUSCRIPT / "main.pdf"
ARCHIVE = MANUSCRIPT / "arxiv-source-gctr.zip"
MANIFEST = MANUSCRIPT / "portfolio_artifacts_manifest.json"
EXPECTED_TITLE = (
    "Query-Efficient Quantum Approximate Optimization "
    "via Graph-Conditioned Trust Regions"
)
EXPECTED_SECTIONS = [
    "Introduction",
    "Background",
    "Related Work",
    "Failure Modes of Structural Optimizer Routing",
    "Risk-Controlled Graph-Conditioned Trust Regions",
    "Experiments",
    "Conclusion",
]
EXPECTED_SUBSECTIONS = [0, 3, 0, 3, 4, 3, 0]
EXPECTED_FIGURES = 13
EXPECTED_TABLES = 5
EXPECTED_PROOFS = 5
EXPECTED_BIBLIOGRAPHY_ENTRIES = 66
EXPECTED_ARCHIVE_MEMBERS = 23
EXPECTED_SOURCE_MANIFESTS = {
    "regular_development": (
        ROOT
        / "portfolio_results/development_v2_tqa_family_gate/portfolio_manifest.json"
    ),
    "heterogeneous_development": (
        ROOT
        / (
            "portfolio_results/heterogeneous_development_v2_tqa_family_gate/"
            "portfolio_manifest.json"
        )
    ),
    "confirmatory": (
        ROOT
        / "portfolio_results/heterogeneous_confirmatory_v1/portfolio_manifest.json"
    ),
}
BANNED_TEXT = {
    "omitted thanks section": r"\backnowledg(?:e)?ments?\b",
    "omitted societal section": r"\bimpact\s+" + r"statement\b",
    "incomplete-proof marker": r"\bproof\s+" + r"sketch\b",
    "forbidden arXiv identifier": r"\b2607\." + r"06758\b",
    "forbidden citation key": r"\bsmith2026" + r"adaptive\b",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def without_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def normalized_tex_text(value: str) -> str:
    value = value.replace(r"\textbf{", "")
    value = value.replace("\\\\", " ").replace("{", "").replace("}", "")
    return " ".join(value.split())


def normalized_title(source: str) -> str:
    match = re.search(r"\\title\{(.*?)\}\s*\\author", source, flags=re.S)
    if not match:
        return ""
    return normalized_tex_text(match.group(1))


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


def canonical_manifest_sha256(manifest: dict[str, object]) -> str:
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def manuscript_path(name: str) -> Path:
    relative = Path(name)
    require(not relative.is_absolute(), f"absolute include path is forbidden: {name}")
    require(".." not in relative.parts, f"parent traversal is forbidden: {name}")
    path = MANUSCRIPT / relative
    require(
        path.resolve().is_relative_to(MANUSCRIPT.resolve()),
        f"include escapes manuscript directory: {name}",
    )
    return path


def included_sources(source: str) -> dict[str, Path]:
    source = without_comments(source)
    relative = (
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source)
        + re.findall(r"\\input\{([^{}]+)\}", source)
    )
    require(len(relative) == len(set(relative)), "an included asset is referenced twice")
    return {name: manuscript_path(name) for name in sorted(relative)}


def structural_report(source: str) -> dict[str, object]:
    split = source.find(r"\appendix")
    require(split >= 0, "appendix transition is missing")
    main = source[:split]
    appendix = source[split:]
    section_matches = list(re.finditer(r"^\\section\{([^{}]+)\}", main, flags=re.M))
    sections = [match.group(1) for match in section_matches]
    require(sections == EXPECTED_SECTIONS, f"main sections differ: {sections}")

    subsection_counts: list[int] = []
    for index, match in enumerate(section_matches):
        end = (
            section_matches[index + 1].start()
            if index + 1 < len(section_matches)
            else len(main)
        )
        subsection_counts.append(
            len(re.findall(r"^\\subsection\{", main[match.start():end], flags=re.M))
        )
    require(
        subsection_counts == EXPECTED_SUBSECTIONS,
        f"main subsection counts differ: {subsection_counts}",
    )

    require(r"\appendix" not in main, "appendix command appears in the main text")
    require(r"\appendix" in appendix, "appendix command is missing")
    require(
        main.count(r"\section*{Limitations}") == 1,
        "exactly one unnumbered Limitations section is required",
    )
    conclusion_start = section_matches[-1].start()
    require(
        r"\section*{Limitations}" in main[conclusion_start:],
        "Limitations must appear after the Conclusion heading",
    )
    require(
        main.count(r"\section*{Data Availability}") == 1
        and main.count(r"\section*{Code Availability}") == 1,
        "unnumbered Data and Code Availability sections are required",
    )

    figure_paths = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source
    )
    table_inputs = [
        name
        for name in re.findall(r"\\input\{([^{}]+)\}", source)
        if name != "tables/portfolio_numbers.tex"
    ]
    macro_inputs = [
        name
        for name in re.findall(r"\\input\{([^{}]+)\}", source)
        if name == "tables/portfolio_numbers.tex"
    ]
    figure_floats = len(re.findall(r"\\begin\{figure\*?\}", source))
    table_floats = len(re.findall(r"\\begin\{table\*?\}", source))
    require(
        len(figure_paths) == len(set(figure_paths)) == EXPECTED_FIGURES,
        "expected 13 distinct referenced figures",
    )
    require(
        figure_floats == EXPECTED_FIGURES,
        f"expected {EXPECTED_FIGURES} figure environments, found {figure_floats}",
    )
    require(
        all(
            Path(name).parent == Path("figures")
            and Path(name).suffix.lower() == ".pdf"
            for name in figure_paths
        ),
        "every manuscript figure must be a PDF below figures/",
    )
    require(
        len(table_inputs) == len(set(table_inputs)) == EXPECTED_TABLES,
        "expected five distinct table inputs",
    )
    require(
        table_floats == EXPECTED_TABLES,
        f"expected {EXPECTED_TABLES} table environments, found {table_floats}",
    )
    require(
        all(
            Path(name).parent == Path("tables")
            and Path(name).suffix.lower() == ".tex"
            for name in table_inputs
        ),
        "every study table must be a TeX input below tables/",
    )
    require(
        macro_inputs == ["tables/portfolio_numbers.tex"],
        "portfolio_numbers.tex must be included exactly once",
    )
    require(r"\onecolumngrid" not in source, "manual one-column override is forbidden")
    require(r"\begin{algorithm}" not in source, "custom algorithm environment is forbidden")

    proof_bodies = re.findall(
        r"\\begin\{proof\}(.*?)\\end\{proof\}", source, flags=re.S
    )
    require(
        source.count(r"\begin{proof}") == source.count(r"\end{proof}")
        == len(proof_bodies) == EXPECTED_PROOFS,
        f"expected exactly {EXPECTED_PROOFS} complete proof environments",
    )
    proof_words = [word_count(body) for body in proof_bodies]
    require(
        all(count >= 75 for count in proof_words),
        f"a proof is too short to be complete: {proof_words}",
    )
    return {
        "main_sections": len(sections),
        "subsections": subsection_counts,
        "figures": figure_floats,
        "figure_paths": figure_paths,
        "tables": table_floats,
        "table_inputs": table_inputs,
        "proofs": len(proof_bodies),
        "proof_word_counts": proof_words,
    }


def citation_report(source: str, bibliography: str) -> dict[str, int]:
    bib_keys = re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", bibliography, flags=re.M)
    require(len(bib_keys) == len(set(bib_keys)), "duplicate bibliography keys")
    require(
        len(bib_keys) == EXPECTED_BIBLIOGRAPHY_ENTRIES,
        f"bibliography has {len(bib_keys)} entries, "
        f"expected {EXPECTED_BIBLIOGRAPHY_ENTRIES}",
    )
    groups = re.findall(
        r"\\cite(?:p|t|alp|alt|author|year|yearpar)?\*?"
        r"(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}",
        source,
    )
    require(60 <= len(groups) <= 70, f"manuscript has {len(groups)} citation commands")
    parsed_groups = [
        [key.strip() for key in group.split(",") if key.strip()]
        for group in groups
    ]
    group_sizes = [len(group) for group in parsed_groups]
    require(all(1 <= size <= 3 for size in group_sizes), "citation group exceeds three works")
    require(
        all(len(group) == len(set(group)) for group in parsed_groups),
        "a citation command repeats a bibliography key",
    )
    cited = {key for group in parsed_groups for key in group}
    missing = sorted(cited - set(bib_keys))
    unused = sorted(set(bib_keys) - cited)
    require(not missing, f"citation keys are missing from bibliography: {missing}")
    require(not unused, f"bibliography contains uncited entries: {unused}")
    require(
        len(cited) == EXPECTED_BIBLIOGRAPHY_ENTRIES,
        f"expected 65 unique cited works, found {len(cited)}",
    )
    require(r"\nocite" not in source, "nocite is not permitted")
    return {
        "bibliography_entries": len(bib_keys),
        "citation_commands": len(groups),
        "citation_mentions": sum(group_sizes),
        "unique_cited_works": len(cited),
    }


def validate_manifest(source: str) -> dict[str, object]:
    require(MANIFEST.is_file(), "portfolio artifact manifest is missing")
    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("schema_version") == 2, "artifact manifest is not schema 2")
    require(
        manifest.get("generator") == "manuscript/generate_portfolio_artifacts.py",
        "artifact manifest generator differs",
    )
    stored_manifest_hash = manifest.get("manifest_sha256")
    require(
        isinstance(stored_manifest_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", stored_manifest_hash) is not None,
        "artifact manifest self-hash is malformed",
    )
    require(
        canonical_manifest_sha256(manifest) == stored_manifest_hash,
        "artifact manifest self-hash differs",
    )

    contract = manifest.get("asset_contract")
    require(isinstance(contract, dict), "artifact contract is missing")
    require(contract.get("figure_count") == EXPECTED_FIGURES, "figure contract differs")
    require(contract.get("figure_formats") == ["pdf", "png"], "figure formats differ")
    figure_stems = contract.get("figure_stems")
    require(
        isinstance(figure_stems, list)
        and len(figure_stems) == len(set(figure_stems)) == EXPECTED_FIGURES,
        "figure stems differ",
    )
    referenced_figures = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source
    )
    require(
        referenced_figures
        == [f"figures/{stem}.pdf" for stem in figure_stems],
        "manuscript figure order differs from the artifact contract",
    )

    require(contract.get("table_count") == EXPECTED_TABLES, "table contract differs")
    table_files = contract.get("table_files")
    require(
        isinstance(table_files, list)
        and len(table_files) == len(set(table_files)) == EXPECTED_TABLES,
        "table file contract differs",
    )
    referenced_tables = [
        f"manuscript/{name}"
        for name in re.findall(r"\\input\{([^{}]+)\}", source)
        if name != "tables/portfolio_numbers.tex"
    ]
    require(
        set(referenced_tables) == set(table_files),
        "referenced study tables differ from the artifact contract",
    )
    require(
        contract.get("all_quantitative_figures_derive_from_validated_locked_evidence")
        is True,
        "locked-evidence declaration is missing",
    )

    source_manifests = manifest.get("source_manifests")
    require(
        isinstance(source_manifests, dict)
        and set(source_manifests) == set(EXPECTED_SOURCE_MANIFESTS),
        "source-manifest set differs",
    )
    for name, path in EXPECTED_SOURCE_MANIFESTS.items():
        require(path.is_file(), f"source portfolio manifest is missing: {path}")
        recorded = json.loads(path.read_text()).get("decision_sha256")
        require(
            source_manifests[name] == recorded,
            f"source decision hash differs for {name}",
        )
    require(
        manifest.get("confirmatory_decision_sha256")
        == source_manifests["confirmatory"],
        "confirmatory decision hash differs",
    )

    files = manifest.get("files", {})
    require(isinstance(files, dict), "artifact file map is missing")
    expected_files = {
        f"manuscript/figures/{stem}.{extension}"
        for stem in figure_stems
        for extension in ("pdf", "png")
    }
    expected_files.update(table_files)
    expected_files.add("manuscript/tables/portfolio_numbers.tex")
    require(
        set(files) == expected_files,
        "artifact file map differs from the schema-2 asset contract",
    )
    for relative, expected in files.items():
        require(
            isinstance(expected, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"manifest hash is malformed: {relative}",
        )
        path = ROOT / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(sha256(path) == expected, f"manifest hash differs: {relative}")
    return {
        "manifest_schema": manifest["schema_version"],
        "manifest_files": len(files),
        "manifest_sha256": stored_manifest_hash,
    }


def archive_sources(source: str) -> dict[str, Path]:
    sources = {
        "main.tex": MAIN,
        "refs.bib": BIB,
        "main.bbl": BBL,
        "prx-popular-summary.tex": POPULAR,
    }
    sources.update(included_sources(source))
    require(
        len(sources) == EXPECTED_ARCHIVE_MEMBERS,
        f"expected {EXPECTED_ARCHIVE_MEMBERS} transitive archive inputs, "
        f"found {len(sources)}",
    )
    return dict(sorted(sources.items()))


def validate_archive(source: str, required: bool) -> dict[str, object]:
    if not required:
        return {"archive_checked": False}
    require(ARCHIVE.is_file(), "submission archive is missing")
    sources = archive_sources(source)
    with zipfile.ZipFile(ARCHIVE) as handle:
        names = handle.namelist()
        require(names == sorted(sources), "submission archive members or order differ")
        require(len(names) == len(set(names)), "submission archive has duplicate members")
        for name, path in sources.items():
            require(path.is_file(), f"archive input is missing: {name}")
            require(handle.read(name) == path.read_bytes(), f"archive member is stale: {name}")
    return {
        "archive_checked": True,
        "archive_members": len(sources),
        "archive_sha256": sha256(ARCHIVE),
    }


def validate(*, require_archive: bool = True) -> dict[str, object]:
    require(MAIN.is_file(), "main.tex is missing")
    require(BIB.is_file(), "refs.bib is missing")
    require(BBL.is_file(), "main.bbl is missing")
    require(POPULAR.is_file(), "PRX popular summary is missing")

    raw_tex = MAIN.read_text()
    bib = BIB.read_text()
    for label, pattern in BANNED_TEXT.items():
        require(
            re.search(pattern, raw_tex + "\n" + bib, flags=re.I) is None,
            f"forbidden text is present: {label}",
        )
    tex = without_comments(raw_tex)
    documentclass = re.search(
        r"\\documentclass(?:\[([^\]]*)\])?\{([^{}]+)\}", tex
    )
    require(documentclass is not None, "document class is missing")
    options = tuple(
        option.strip()
        for option in (documentclass.group(1) if documentclass else "").split(",")
        if option.strip()
    )
    require(
        documentclass is not None
        and documentclass.group(2) == "revtex4-2"
        and options == ("aps", "prx", "reprint", "superscriptaddress", "floatfix"),
        "exact PRX REVTeX document class is missing",
    )
    require(
        r"\bibliographystyle{apsrev4-2}" in tex,
        "APS bibliography style is missing",
    )
    require(
        r"\renewcommand{\thesection}" not in tex
        and r"\renewcommand{\thesubsection}" not in tex,
        "manual section-number overrides are forbidden",
    )
    require(normalized_title(tex) == EXPECTED_TITLE, "visible title differs")
    metadata_title = re.search(r"pdftitle\s*=\s*\{([^{}]+)\}", tex)
    require(
        metadata_title is not None
        and normalized_tex_text(metadata_title.group(1)) == EXPECTED_TITLE,
        "PDF metadata title differs",
    )
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    require(abstract is not None, "abstract is missing")
    abstract_words = word_count(abstract.group(1)) if abstract else 0
    require(100 <= abstract_words <= 250, f"abstract has {abstract_words} words")
    popular_words = word_count(POPULAR.read_text())
    require(
        50 <= popular_words <= 150,
        f"PRX popular summary has {popular_words} words",
    )
    require(r"\bibliography{refs}" in tex, "refs.bib is not the manuscript bibliography")

    included = included_sources(tex)
    for name, path in included.items():
        require(path.is_file(), f"included source is missing: {name}")
    labels = re.findall(r"\\label\{([^{}]+)\}", tex)
    require(len(labels) == len(set(labels)), "duplicate LaTeX labels")
    refs = re.findall(r"\\(?:eq)?ref\{([^{}]+)\}", tex)
    require(not (set(refs) - set(labels)), "reference points to a missing label")
    require(PDF.is_file() and PDF.stat().st_size > 100_000, "compiled PDF is missing")
    bbl_keys = re.findall(
        r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
        BBL.read_text(),
        flags=re.S,
    )
    require(
        len(bbl_keys) == len(set(bbl_keys)) == EXPECTED_BIBLIOGRAPHY_ENTRIES,
        "compiled bibliography does not contain 66 unique entries",
    )

    report = {
        "main_sha256": sha256(MAIN),
        "bibliography_sha256": sha256(BIB),
        "bbl_sha256": sha256(BBL),
        "abstract_words": abstract_words,
        "popular_summary_words": popular_words,
        "included_sources": len(included),
        **structural_report(tex),
        **citation_report(tex, bib),
        **validate_manifest(tex),
        **validate_archive(tex, require_archive),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        validate(require_archive=not args.skip_archive),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
