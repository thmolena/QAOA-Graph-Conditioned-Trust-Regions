"""`gctr-reproduce` console entrypoint.

Regenerates every DATA figure, table and source-data CSV in the manuscript
from an end-to-end simulation run: exact-statevector QAOA on MaxCut, a trained torch
GNN predicting a graph-conditioned Gaussian over angles, a held-out error
calibrator, and query-efficiency / calibration / cross-size /
leave-one-family-out / ablation / seed-stability / shot-noise studies. Nothing
is maintained separately; algorithmic outputs are deterministic functions of the seed
in a fixed software environment (the environment used for the committed run is
recorded in meta.json; wall-clock runtime columns are machine-dependent by
nature). Data figures and deterministic conceptual schematics are all rebuilt
by the replot path.

Modes
-----
default          : full run (the manuscript configuration).
--quick          : reduced configuration (smoke test, ~1 min).
--replot-only    : skip the simulation entirely; re-render every figure and
                   table from the committed source_data files (seconds).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .pipeline import (Config, METHOD_ORDER, prepare, train_model,
                       evaluate_queries, evaluate_calibration,
                       evaluate_generalization, evaluate_lofo,
                       evaluate_ablation, evaluate_seed_stability,
                       evaluate_shot_noise, landscape_slice, pipeline_example)
from .plots import build_all
from .tables import write_tables
from . import __version__


SCHEMA_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_files() -> list[Path]:
    """Return the Python sources that define the installable implementation.

    Package metadata is versioned separately.  Restricting this list to files
    shipped inside ``specops_gctr`` makes the digest identical in a source
    checkout and an installed wheel.
    """
    package_dir = Path(__file__).resolve().parent
    # Portfolio-v1 is a separate prospective experiment and does not alter the
    # frozen schema-2 manuscript generator/replot contract.  Excluding those
    # additive modules preserves the meaning of the committed legacy manifest;
    # the portfolio runner records its own broader implementation digest.
    portfolio_only = {
        "fixed_budget.py", "portfolio.py", "portfolio_experiment.py",
        "protocol.py",
    }
    return sorted((path for path in package_dir.rglob("*.py")
                   if path.name not in portfolio_only),
                  key=lambda path: path.relative_to(package_dir).as_posix())


def _implementation_fingerprint() -> str:
    """Hash the installable Python implementation with wheel-stable paths."""
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in _implementation_files():
        relative = Path(package_dir.name) / path.relative_to(package_dir)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_path(root: Path, recorded_path: str) -> Path:
    """Resolve a portable manifest path against the manuscript root."""
    path = Path(recorded_path)
    return path if path.is_absolute() else root / path


def validate_manifest(manuscript_dir: Path, figures_dir: Path | None = None) -> dict:
    """Validate a completed run against its recorded hashes and source digest.

    Raises ``RuntimeError`` on any mismatch and returns the parsed manifest on
    success.  This is intentionally cheap: no simulation or plotting occurs.
    """
    root = Path(manuscript_dir).resolve()
    figures = Path(figures_dir) if figures_dir else root / "figures"
    manifest_path = figures / "specops_gctr_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "package": "specops-gctr",
        "package_version": __version__,
        "replot_implementation_sha256": _implementation_fingerprint(),
    }
    errors = []
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            errors.append(
                f"manifest {key}: expected {value!r}, got {manifest.get(key)!r}")

    source_dir = _manifest_path(root, manifest.get("source_data", "source_data"))
    source_records = []
    for name in ("meta.json", "results.json"):
        path = source_dir / name
        if not path.is_file():
            errors.append(f"missing source record: {path}")
            continue
        record = json.loads(path.read_text())
        source_records.append((name, record))
        if record.get("package") != "specops-gctr":
            errors.append(f"{name} does not identify package specops-gctr")
        if record.get("schema_version") != manifest.get("source_schema_version"):
            errors.append(
                f"{name} schema_version does not match source_schema_version")

    if len(source_records) == 2:
        recorded_hashes = [
            record.get("implementation_sha256",
                       record.get("generator_implementation_sha256"))
            for _, record in source_records
        ]
        if recorded_hashes[0] != recorded_hashes[1]:
            errors.append("meta.json and results.json disagree on generator hash")
        if recorded_hashes[0] is not None and manifest.get(
                "generator_implementation_sha256") != recorded_hashes[0]:
            errors.append("manifest generator hash does not match source records")
        semantics = [record.get("score_semantics")
                     for _, record in source_records]
        if semantics[0] != semantics[1]:
            errors.append("meta.json and results.json disagree on score semantics")
        if manifest.get("score_semantics") != semantics[0]:
            errors.append("manifest score semantics do not match source records")
        attainment = [record.get("target_attainment_recorded")
                      for _, record in source_records]
        if attainment[0] != attainment[1]:
            errors.append(
                "meta.json and results.json disagree on attainment provenance")
        if manifest.get("target_attainment_recorded") != attainment[0]:
            errors.append(
                "manifest attainment provenance does not match source records")

    for section in ("source_data_sha256", "generated_artifact_sha256"):
        hashes = manifest.get(section)
        if not isinstance(hashes, dict) or not hashes:
            errors.append(f"missing or empty {section}")
            continue
        for recorded_path, expected_hash in hashes.items():
            path = _manifest_path(root, recorded_path)
            if not path.is_file():
                errors.append(f"missing hashed file: {path}")
            else:
                if (section == "generated_artifact_sha256"
                        and path.suffix.lower() == ".pdf"):
                    payload = path.read_bytes()
                    if b"/CreationDate" in payload or b"/ModDate" in payload:
                        errors.append(
                            f"volatile date metadata in generated PDF: "
                            f"{recorded_path}")
                actual_hash = _sha256(path)
                if actual_hash != expected_hash:
                    errors.append(
                        f"hash mismatch for {recorded_path}: "
                        f"expected {expected_hash}, got {actual_hash}")
    if errors:
        raise RuntimeError("manifest validation failed:\n- " + "\n- ".join(errors))
    return manifest


def _default_manuscript_dir() -> Path:
    """Locate the manuscript root (the directory holding main.tex) relative to
    the package. In a repository checkout the package lives at
    ``<repo>/specops_gctr`` and the manuscript at ``<repo>/manuscript``;
    for a pip-installed package with no repository nearby, fall back to the
    current working directory (outputs land in ./source_data, ./figures,
    ./tables)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "main.tex").is_file():
            return parent
        if (parent / "manuscript" / "main.tex").is_file():
            return parent / "manuscript"
    return Path.cwd()


def _environment() -> dict:
    """Software environment fingerprint recorded next to the committed run."""
    import matplotlib
    import networkx
    import pandas
    import scipy
    import torch
    return dict(python=platform.python_version(),
                platform=platform.platform(),
                machine=platform.machine(),
                numpy=np.__version__, scipy=scipy.__version__,
                networkx=networkx.__version__, torch=torch.__version__,
                matplotlib=matplotlib.__version__, pandas=pandas.__version__,
                torch_num_threads=torch.get_num_threads())


def write_source_data(src: Path, cfg: Config, summary, calib, gen, lofo, abl,
                      seeds, shot, land, pipe_ex):
    src.mkdir(parents=True, exist_ok=True)

    with open(src / "Figure2_QueryEfficiency.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["method", "evaluations", "evaluations_sd",
                    "evaluations_used", "evaluations_used_sd",
                    "successes", "n_test",
                    "expectation_ratio", "expectation_ratio_sd",
                    "sampled_ratio", "sampled_ratio_sd",
                    "runtime_ms", "runtime_ms_sd", "wilcoxon_p_vs_gctr"])
        for m in METHOD_ORDER:
            s = summary[m]
            p = summary["_wilcoxon"].get(m, {}).get("p", "")
            # two decimals on evals/runtime so downstream :.0f formatting in
            # tables.py does not double-round (e.g. 13.475 -> 13.5 -> 14)
            w.writerow([m, round(s["evals_mean"], 2), round(s["evals_sd"], 2),
                        round(s["evaluations_used_mean"], 2),
                        round(s["evaluations_used_sd"], 2),
                        s["successes"], len(s["reached_target"]),
                        round(s["expratio_mean"], 3), round(s["expratio_sd"], 3),
                        round(s["ratio_mean"], 3), round(s["ratio_sd"], 3),
                        round(s["ms_mean"], 2), round(s["ms_sd"], 2),
                        (round(p, 6) if p != "" else "")])

    with open(src / "QueryEfficiency_InstanceLevel.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["instance_index", "family", "graph_seed", "n", "maxcut",
                    "target", "method", "capped_score", "evaluations_used",
                    "reached_target", "expectation_ratio", "sampled_ratio"])
        for instance in summary["_instances"]:
            i = instance["index"]
            for method in METHOD_ORDER:
                record = summary[method]
                w.writerow([
                    i, instance["family"], instance["graph_seed"], instance["n"],
                    instance["maxcut"], f'{instance["target"]:.12g}', method,
                    record["evals"][i], record["evaluations_used"][i],
                    record["reached_target"][i],
                    f'{record["expectation_ratios"][i]:.12g}',
                    f'{record["sampled_ratios"][i]:.12g}',
                ])

    with open(src / "Figure3_CalibrationAndUncertainty.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["nominal_coverage", "observed_coverage", "covered_count"])
        c = calib["curve"]
        for p, o, cnt in zip(c["predicted"], c["observed"], c["counts"]):
            w.writerow([round(p, 4), round(o, 4), cnt])

    with open(src / "Figure4_Generalization.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "random_evals", "heuristic_evals", "point_evals",
                    "gctr_evals", "speedup_vs_random", "random_successes",
                    "heuristic_successes", "point_successes",
                    "gctr_successes", "n_test"])
        for n in sorted(gen.keys()):
            g = gen[n]
            w.writerow([n, round(g["random_evals"], 1),
                        round(g["heuristic_evals"], 1),
                        round(g["point_evals"], 1),
                        round(g["gctr_evals"], 1), round(g["speedup"], 2),
                        g["random_successes"], g["heuristic_successes"],
                        g["point_successes"], g["gctr_successes"], g["n_test"]])

    with open(src / "Figure5_Ablation.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "evaluations", "evaluations_sd", "successes",
                    "n_test", "ece", "spearman", "spearman_p"])
        for m, s in abl.items():
            w.writerow([m, round(s["evals_mean"], 2), round(s["evals_sd"], 2),
                        s["successes"], s["n_test"],
                        round(s["ece"], 3), round(s["spearman"], 3),
                        f"{s['spearman_p']:.8g}"])

    with open(src / "Figure7_BudgetPolicy.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["z", "K_seeds", "T_budget_cap"])
        al = summary["_allocation"]
        for z, kk, tt in zip(al["z"], al["K"], al["T"]):
            w.writerow([round(z, 4), kk, tt])

    with open(src / "LOFO_InstanceLevel.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["held_out_family", "graph_seed", "target", "method",
                    "capped_score", "evaluations_used", "reached_target"])
        lofo_methods = ("GCTR", "GNN point", "Heuristic",
                        "GCTR no trust region", "GCTR two-seed/full-cap")
        for row in lofo["rows"]:
            for method in lofo_methods:
                record = row[method]
                w.writerow([
                    row["held_out_family"], row["graph_seed"],
                    f'{row["target"]:.12g}', method, record["capped_score"],
                    record["evaluations_used"], record["reached_target"],
                ])

    with open(src / "EDFig2_ShotNoise.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["shots", "method", "evaluations", "evaluations_sd",
                    "successes", "n_test", "expectation_ratio",
                    "expectation_ratio_sd"])
        for shots in sorted(shot.keys()):
            for m, s in shot[shots].items():
                w.writerow([shots, m, round(s["evals_mean"], 1),
                            round(s["evals_sd"], 1), s["successes"], s["n_test"],
                            round(s["expratio_mean"], 3),
                            round(s["expratio_sd"], 3)])

    with open(src / "EDFig3_SeedStability.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["seed", "method", "evaluations", "expectation_ratio"])
        for s in seeds["seeds"]:
            q = seeds["per_seed"][str(s)]["queries"]
            for m in METHOD_ORDER:
                w.writerow([s, m, round(q[m]["evals_mean"], 1),
                            round(q[m]["expratio_mean"], 3)])

    speeds = [gen[n]["speedup"] for n in gen]
    agg = seeds["aggregate"]
    readouts = {
        "evals::GCTR": round(summary["GCTR"]["evals_mean"], 1),
        "evals_sd::GCTR": round(summary["GCTR"]["evals_sd"], 1),
        "evals::Random": round(summary["Random"]["evals_mean"], 1),
        "evals::Heuristic": round(summary["Heuristic"]["evals_mean"], 1),
        "evals::GNN point": round(summary["GNN point"]["evals_mean"], 1),
        "expratio::GCTR": round(summary["GCTR"]["expratio_mean"], 3),
        "reduction_vs_random": round(
            1.0 - summary["GCTR"]["evals_mean"] / summary["Random"]["evals_mean"], 3),
        "speedup_min": round(min(speeds), 2),
        "speedup_max": round(max(speeds), 2),
        "ece": round(calib["ece"], 3),
        "spearman": round(calib["spearman"], 3),
        "spearman_p": float(f'{calib["spearman_p"]:.8g}'),
        "trsigma_spread": round(calib["trsigma_spread"], 5),
        "lofo_gctr": round(lofo["pooled"]["gctr_evals_mean"], 1),
        "lofo_point": round(lofo["pooled"]["point_evals_mean"], 1),
        "lofo_heuristic": round(lofo["pooled"]["heuristic_evals_mean"], 1),
        "lofo_p_vs_heuristic": float(
            f'{lofo["wilcoxon"]["Heuristic"]["p"]:.8g}'),
        "lofo_p_vs_point": float(
            f'{lofo["wilcoxon"]["GNN point"]["p"]:.8g}'),
        "seeds_gctr": f'{agg["GCTR"]["evals_mean"]:.1f}+-{agg["GCTR"]["evals_sd"]:.1f}',
        "seeds_heuristic": (f'{agg["Heuristic"]["evals_mean"]:.1f}'
                            f'+-{agg["Heuristic"]["evals_sd"]:.1f}'),
        "seeds_spearman": (f'{agg["_spearman"]["mean"]:.3f}'
                           f'+-{agg["_spearman"]["sd"]:.3f}'),
        "gctr_successes": summary["GCTR"]["successes"],
        "gctr_n_test": len(summary["GCTR"]["reached_target"]),
        "gctr_max_evaluations_used": max(summary["GCTR"]["evaluations_used"]),
        "gctr_max_allocated_cap": max(summary["_allocation"]["T"]),
    }
    meta = dict(schema_version=SCHEMA_VERSION, package="specops-gctr",
                package_version=__version__, bib_key="huynh2026gctr",
                implementation_sha256=_implementation_fingerprint(),
                score_semantics="capped_cost_to_shared_target",
                target_attainment_recorded=True,
                config=cfg.to_dict(), environment=_environment(),
                n_test=4 * cfg.per_family_test,
                ece=calib["ece"], spearman=calib["spearman"],
                spearman_p=calib["spearman_p"],
                n_residuals=calib["curve"]["N"],
                lofo=lofo, landscape=land, pipeline_example=pipe_ex,
                readouts=readouts)
    with open(src / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return readouts


def run(argv=None):
    ap = argparse.ArgumentParser(
        prog="gctr-reproduce",
        description="Regenerate every data figure, table and CSV of the GCTR "
                    "manuscript from a recorded run (or replot from committed "
                    "source data).")
    ap.add_argument("--version", action="version",
                    version=f"specops-gctr {__version__}")
    ap.add_argument("--manuscript-dir", default=None,
                    help="manuscript root (default: auto-located)")
    ap.add_argument("--source-data-dir", default=None)
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--tables-dir", default=None)
    ap.add_argument("--seed", type=int, default=20260424)
    ap.add_argument("--quick", action="store_true",
                    help="reduced configuration (smoke test, ~1 min)")
    ap.add_argument("--replot-only", action="store_true",
                    help="re-render figures/tables from committed CSVs only")
    ap.add_argument("--validate-only", action="store_true",
                    help="validate the committed manifest and hashes; do no work")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    mroot = Path(args.manuscript_dir) if args.manuscript_dir \
        else _default_manuscript_dir()
    src = Path(args.source_data_dir) if args.source_data_dir \
        else mroot / "source_data"
    figs = Path(args.figures_dir) if args.figures_dir else mroot / "figures"
    tabs = Path(args.tables_dir) if args.tables_dir else mroot / "tables"

    if args.validate_only:
        manifest = validate_manifest(mroot, figs)
        print("[valid] replot implementation, source records and hashes match")
        return manifest

    if not args.replot_only:
        cfg = Config(seed=args.seed)
        if args.quick:
            cfg.train_n = 10
            cfg.per_family_train = 3
            cfg.per_family_val = 2
            cfg.per_family_test = 2
            cfg.epochs = 40
            cfg.cross_sizes = (8, 10, 12)
            cfg.stability_seeds = (args.seed, 1)
            cfg.shot_levels = (256,)
            cfg.n_target_starts = 4
        print("[config]", json.dumps(cfg.to_dict()))
        data = prepare(cfg)
        model, _, _ = train_model(cfg, data)
        summary, mu, var = evaluate_queries(cfg, data, model)
        calib = evaluate_calibration(cfg, data, model)
        gen = evaluate_generalization(cfg, data, model)
        lofo = evaluate_lofo(cfg, data)
        abl = evaluate_ablation(cfg, data)
        seeds = evaluate_seed_stability(cfg, data)
        shot = evaluate_shot_noise(cfg, data, model)
        land = landscape_slice(cfg, data, model)
        pipe_ex = pipeline_example(cfg, data, model)
        readouts = write_source_data(src, cfg, summary, calib, gen, lofo, abl,
                                     seeds, shot, land, pipe_ex)
        results = dict(schema_version=SCHEMA_VERSION,
                       package="specops-gctr", package_version=__version__,
                       implementation_sha256=_implementation_fingerprint(),
                       score_semantics="capped_cost_to_shared_target",
                       target_attainment_recorded=True,
                       config=cfg.to_dict(), environment=_environment(),
                       queries=summary, calibration=calib, generalization=gen,
                       lofo=lofo, ablation=abl, seed_stability=seeds,
                       shot_noise=shot, readouts=readouts)
        (src / "results.json").write_text(json.dumps(results, indent=2))
        print("[sim] wrote source_data CSVs + meta.json + results.json to", src)

    figures = build_all(src, figs, formats=tuple(args.formats), dpi=args.dpi)
    tables = write_tables(src, tabs)

    # Keep the committed manifest portable. Plot/table helpers return the paths
    # they wrote, which may be absolute when the command is launched with an
    # absolute manuscript directory. Recording those paths would leak a local
    # username and make byte-for-byte audits depend on the checkout location.
    root = mroot.resolve()

    def portable(path):
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return Path(path).as_posix()

    with open(src / "meta.json") as f:
        source_meta = json.load(f)
    manifest = dict(schema_version=SCHEMA_VERSION, package="specops-gctr",
                    package_version=__version__, bib_key="huynh2026gctr",
                    replot_implementation_sha256=_implementation_fingerprint(),
                    source_schema_version=source_meta.get("schema_version", 1),
                    score_semantics=source_meta.get(
                        "score_semantics", "capped_cost_to_shared_target"),
                    target_attainment_recorded=source_meta.get(
                        "target_attainment_recorded", True),
                    mode="replot-only" if args.replot_only else
                    ("quick" if args.quick else "full"),
                    source_data=portable(src),
                    figures=[portable(path) for path in figures],
                    tables=[portable(path) for path in tables])
    manifest["readouts"] = source_meta.get("readouts", {})
    if "legacy_readout_note" in source_meta:
        manifest["legacy_readout_note"] = source_meta["legacy_readout_note"]
    recorded_generator = source_meta.get(
        "implementation_sha256",
        source_meta.get("generator_implementation_sha256"),
    )
    if recorded_generator is not None:
        manifest["generator_implementation_sha256"] = recorded_generator
    source_files = sorted(path for path in Path(src).iterdir() if path.is_file())
    manifest["source_data_sha256"] = {
        portable(path): _sha256(path) for path in source_files}
    manifest["generated_artifact_sha256"] = {
        portable(path): _sha256(path)
        for path in [*map(Path, figures), *map(Path, tables)]}
    Path(figs).mkdir(parents=True, exist_ok=True)
    (Path(figs) / "specops_gctr_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"[done] {len(figures)} figure files -> {figs}")
    print(f"[done] {len(tables)} tables -> {tabs}")
    print("[readouts]", json.dumps(manifest["readouts"], indent=2))
    return manifest


def main(argv=None) -> int:
    """Console-script entry point: returns a process exit code (0 on success)."""
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
