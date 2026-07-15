"""`gctr-reproduce` console entrypoint.

Regenerates every DATA figure, table and source-data CSV in the manuscript
from a real end-to-end run: exact-statevector QAOA on MaxCut, a trained torch
GNN predicting a graph-conditioned Gaussian over angles, a held-out error
calibrator, and query-efficiency / calibration / cross-size /
leave-one-family-out / ablation / seed-stability / shot-noise studies. Nothing
is hand-typed; all algorithmic outputs are deterministic functions of the seed
in a fixed software environment (the environment used for the committed run is
recorded in meta.json; wall-clock runtime columns are machine-dependent by
nature). Figures 8, 10 and 11 are conceptual schematics and are the only
committed figures not rebuilt here.

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


def _default_manuscript_dir() -> Path:
    """Locate the manuscript root (the directory holding main.tex) relative to
    the package. In a repository checkout the package lives at
    ``<repo>/code/specops_gctr`` and the manuscript at ``<repo>/manuscript``;
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
                    "expectation_ratio", "expectation_ratio_sd",
                    "sampled_ratio", "sampled_ratio_sd",
                    "runtime_ms", "runtime_ms_sd", "wilcoxon_p_vs_gctr"])
        for m in METHOD_ORDER:
            s = summary[m]
            p = summary["_wilcoxon"].get(m, {}).get("p", "")
            # two decimals on evals/runtime so downstream :.0f formatting in
            # tables.py does not double-round (e.g. 13.475 -> 13.5 -> 14)
            w.writerow([m, round(s["evals_mean"], 2), round(s["evals_sd"], 2),
                        round(s["expratio_mean"], 3), round(s["expratio_sd"], 3),
                        round(s["ratio_mean"], 3), round(s["ratio_sd"], 3),
                        round(s["ms_mean"], 2), round(s["ms_sd"], 2),
                        (round(p, 6) if p != "" else "")])

    with open(src / "Figure3_CalibrationAndUncertainty.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["nominal_coverage", "observed_coverage", "covered_count"])
        c = calib["curve"]
        for p, o, cnt in zip(c["predicted"], c["observed"], c["counts"]):
            w.writerow([round(p, 4), round(o, 4), cnt])

    with open(src / "Figure4_Generalization.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "random_evals", "heuristic_evals", "point_evals",
                    "gctr_evals", "speedup_vs_random"])
        for n in sorted(gen.keys()):
            g = gen[n]
            w.writerow([n, round(g["random_evals"], 1),
                        round(g["heuristic_evals"], 1),
                        round(g["point_evals"], 1),
                        round(g["gctr_evals"], 1), round(g["speedup"], 2)])

    with open(src / "Figure5_Ablation.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "evaluations", "evaluations_sd", "ece",
                    "spearman", "spearman_p"])
        for m, s in abl.items():
            w.writerow([m, round(s["evals_mean"], 2), round(s["evals_sd"], 2),
                        round(s["ece"], 3), round(s["spearman"], 3),
                        round(s["spearman_p"], 4)])

    with open(src / "Figure7_BudgetPolicy.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["z", "K_seeds", "T_budget_cap"])
        al = summary["_allocation"]
        for z, kk, tt in zip(al["z"], al["K"], al["T"]):
            w.writerow([round(z, 4), kk, tt])

    with open(src / "EDFig2_ShotNoise.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["shots", "method", "evaluations", "evaluations_sd",
                    "expectation_ratio", "expectation_ratio_sd"])
        for shots in sorted(shot.keys()):
            for m, s in shot[shots].items():
                w.writerow([shots, m, round(s["evals_mean"], 1),
                            round(s["evals_sd"], 1),
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
        "spearman_p": round(calib["spearman_p"], 4),
        "trsigma_spread": round(calib["trsigma_spread"], 5),
        "lofo_gctr": round(lofo["pooled"]["gctr_evals_mean"], 1),
        "lofo_point": round(lofo["pooled"]["point_evals_mean"], 1),
        "lofo_heuristic": round(lofo["pooled"]["heuristic_evals_mean"], 1),
        "seeds_gctr": f'{agg["GCTR"]["evals_mean"]:.1f}+-{agg["GCTR"]["evals_sd"]:.1f}',
        "seeds_heuristic": (f'{agg["Heuristic"]["evals_mean"]:.1f}'
                            f'+-{agg["Heuristic"]["evals_sd"]:.1f}'),
        "seeds_spearman": (f'{agg["_spearman"]["mean"]:.3f}'
                           f'+-{agg["_spearman"]["sd"]:.3f}'),
    }
    meta = dict(package="specops-gctr", bib_key="huynh2026gctr",
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
                    "manuscript from a real run (or replot from committed "
                    "source data).")
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
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    mroot = Path(args.manuscript_dir) if args.manuscript_dir \
        else _default_manuscript_dir()
    src = Path(args.source_data_dir) if args.source_data_dir \
        else mroot / "source_data"
    figs = Path(args.figures_dir) if args.figures_dir else mroot / "figures"
    tabs = Path(args.tables_dir) if args.tables_dir else mroot / "tables"

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
        results = dict(config=cfg.to_dict(), environment=_environment(),
                       queries=summary, calibration=calib, generalization=gen,
                       lofo=lofo, ablation=abl, seed_stability=seeds,
                       shot_noise=shot, readouts=readouts)
        (src / "results.json").write_text(json.dumps(results, indent=2))
        print("[sim] wrote source_data CSVs + meta.json + results.json to", src)

    figures = build_all(src, figs, formats=tuple(args.formats), dpi=args.dpi)
    tables = write_tables(src, tabs)

    manifest = dict(package="specops-gctr", bib_key="huynh2026gctr",
                    mode="replot-only" if args.replot_only else
                    ("quick" if args.quick else "full"),
                    source_data=str(src), figures=figures, tables=tables)
    with open(src / "meta.json") as f:
        manifest["readouts"] = json.load(f).get("readouts", {})
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
