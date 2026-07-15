# Graph-Conditioned Trust Regions for Query-Efficient QAOA

A learned, graph-conditioned **trust-region** method for setting QAOA parameters. On the benchmark below it matches the best fixed-angle heuristic's solution quality while decisively beating every other baseline on cost-to-target (**92.5% fewer objective evaluations than random restarts, ~13×**), and — the payoff — it stays fast under distribution shift, where the fixed heuristic collapses. Its predictive uncertainty is calibrated to realized error and is used to allocate per-instance seeds and budget. The full manuscript and a reproducible code package are included — every figure, table, and number is produced by the code in `code/`.

- **Manuscript:** [`manuscript/main.tex`](manuscript/main.tex)
- **Code (installable):** [`code/`](code) — `pip install ./code`
- **Preprint:** arXiv:2604.24803

## What it does

A graph isomorphism network with Laplacian spectral positional encodings maps a MaxCut instance to a Gaussian over the 2*p* QAOA angles: the mean is a warm start and a **diagonal covariance** sizes a Mahalanobis **trust region** searched by a budgeted local optimizer. Two components make it query-efficient and trustworthy:

- a **held-out error calibrator** — a decoupled head fit on held-out *validation* residuals of the realized objective error — that separates "how large is the trust region" from "how uncertain is the warm start," giving predicted uncertainty that tracks realized error;
- a **heuristic-seeded, self-recentering search with uncertainty-allocated seeds and budget**: it evaluates the concentration-heuristic angles as one of its seeds (so it is never worse than the cheap prior on quality, by construction), re-centers the trust region on the winner, and uses the calibrated uncertainty to set the per-instance seed count and budget cap.

## Install

```bash
pip install ./code            # or:  pip install -e ./code   (editable)
pip install "git+https://github.com/thmolena/QAOA-Graph-Conditioned-Trust-Regions.git#subdirectory=code"
cd code && pip install -e ".[test]" && pytest -q   # run the 19 tests
```

Python ≥ 3.9; dependencies install automatically.

The distribution name is `specops-gctr`. The local-folder and Git URL commands
above work now; bare `pip install specops-gctr` will work only after the built
distribution is uploaded to a Python package index.

## Reproduce every figure and table

```bash
gctr-reproduce                 # full manuscript run (exact-statevector QAOA + trained surrogate), ~30 min on a laptop
gctr-reproduce --quick         # reduced smoke configuration
gctr-reproduce --replot-only   # re-render from committed source data, seconds
```

Runs are deterministic given the seed (default `20260424`) **and the recorded environment** (see the `environment` block in `manuscript/source_data/meta.json`). The model-free baselines (random restarts, concentration heuristic, k-NN, TQA) reproduce bit-for-bit on any machine; the learned-model numbers can shift slightly across torch/BLAS versions, and the `runtime_ms` readouts are machine-dependent.

## Results (as produced by the code in this repository)

At *n* = 14 vertices, depth *p* = 2, on 240 train / 80 validation / 48 test MaxCut instances (four graph families), under a cost-to-98%-target protocol with a shared 400-evaluation cap (canonical seed):

| Method | Evaluations to target (mean ± sd) | Expectation ratio ⟨C⟩/C_max | Evals under family shift (LOFO, pooled) |
| --- | --- | --- | --- |
| Random restarts | 203.9 ± 152.9 | 0.834 | — |
| Concentration heuristic (median training angles) | 12.3 ± 6.4 | 0.857 | 68.1 |
| k-NN transfer | 46.0 ± 117.9 | 0.853 | — |
| TQA schedule | 18.5 ± 13.5 | 0.843 | — |
| Learned point predictor (GNN) | 179.4 ± 125.5 | 0.801 | 118.8 |
| **Graph-conditioned trust region (this work)** | **15.3 ± 8.1** | **0.856** | **39.0** |

- **92.5% fewer objective evaluations** than random-restart search (~13×), with decisive paired Wilcoxon wins (n = 48) over random restarts (p = 3.3e-9), the GNN point predictor (p = 1.6e-9), and k-NN (p = 4.9e-3); vs TQA it is a statistical tie on count (p = 0.091) at higher quality.
- **In distribution the concentration heuristic is leaner** (12.3 vs 15.3 evaluations, p = 5.1e-4 in the heuristic's favor). GCTR evaluates the heuristic's angles as one of its seeds, so it matches the heuristic's quality (0.856 vs 0.857), is never worse on quality by construction, and pays ~3 counted seed evaluations for that insurance.
- **The insurance pays off under distribution shift** (leave-one-family-out): pooled GCTR needs 39.0 evaluations vs 68.1 for the fixed heuristic and 118.8 for the GNN point predictor; on held-out Watts–Strogatz the fixed heuristic collapses to 154.0 while GCTR needs 44.6. The fixed prior is brittle; the graph-conditioned policy adapts.
- **What drives that shift result:** freezing the adaptive allocation raises the pooled count to 58.1 evaluations, while removing the hard Mahalanobis constraint leaves it essentially unchanged at 38.3. The evidence supports the conditioned center and uncertainty-based allocation; it does not isolate an evaluation-count benefit from the constraint itself.
- **Calibrated uncertainty**: coverage ECE 0.049; Spearman ρ = 0.673 between calibrated uncertainty and realized error (p = 1.6e-7, n = 48). The uncertainty is not decorative — it sets per-instance seed counts and budget caps (the budget rule).
- **Cross-size transfer**: trained at *n* = 14, the model gives a 9.6–13.6× evaluation speedup over random restarts at *n* = 8–16.
- **Stability**: across 4 seeds, GCTR needs 16.4 ± 3.0 evaluations (heuristic 12.3 ± 0.0) with ρ = 0.715 ± 0.040; the method ordering is preserved under sampling noise at S = 128 and S = 1024 shots (GCTR 18.5 evals @ 0.854 and 18.0 @ 0.856).

**Scope:** in distribution, the fixed concentration heuristic reaches the target with *fewer* evaluations than this method (12.3 vs 15.3). What the method buys, by seeding with the heuristic's angles, is quality that is never worse than the prior's by construction, plus robustness when the instance family shifts (39.0 vs 68.1 pooled; 44.6 vs 154.0 on held-out Watts–Strogatz). Evaluation is on exact-statevector simulation at *p* = 2, *n* ≤ 16. All of it is visible in the reproduction output.

## Repository layout

```
manuscript/       LaTeX source, figures/, tables/, source_data/
code/             installable package `specops_gctr` v3.0.0 (pip install ./code), tests/
index.html        project page
LICENSE           MIT
CITATION.cff
```

## Cite

```bibtex
@article{huynh2026gctr,
  title   = {Query-Efficient Quantum Approximate Optimization via Graph-Conditioned Trust Regions},
  author  = {Huynh, Molena},
  journal = {arXiv preprint arXiv:2604.24803},
  year    = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Free to use, inspect, modify, and redistribute.
