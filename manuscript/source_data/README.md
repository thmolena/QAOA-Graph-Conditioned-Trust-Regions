# Source data for the manuscript's data figures

Every file in this directory is written by a single run of the released
package (`pip install ./code && gctr-reproduce`); nothing is transcribed by
hand. `meta.json` records the exact configuration and software environment of
the committed run, and `results.json` holds the complete machine-readable
results, including per-instance evaluation counts. Before journal publication,
these files should additionally be deposited in a permanent public repository
such as Zenodo and cited with a DOI.

Algorithmic values are deterministic given the seed (default `20260424`) and
the recorded environment; the `runtime_ms` columns are wall-clock measurements
and vary by machine.

Author: Molena Huynh, North Carolina State University, `molena.huynh@jmp.com`.

## Files

### `Figure2_QueryEfficiency.csv`

Per-method query efficiency at the training size and QAOA depth `p = 2` under
the cost-to-target protocol (stop at 98% of the instance's reference objective
value or at the shared 400-evaluation cap). One row per method.

| Column | Description |
| --- | --- |
| `method` | `Random`, `Heuristic`, `k-NN`, `TQA`, `GNN point`, `GCTR`. |
| `evaluations`, `evaluations_sd` | Mean / SD objective (circuit) evaluations to target. |
| `expectation_ratio`, `expectation_ratio_sd` | Exact QAOA expectation ratio F(θ)/C_max of the returned angles. |
| `sampled_ratio`, `sampled_ratio_sd` | Sampled best-bitstring ratio (512 Born-rule samples; saturates near 1 at these sizes — reported for completeness). |
| `runtime_ms`, `runtime_ms_sd` | Measured wall-clock runtime (machine-dependent). |
| `wilcoxon_p_vs_gctr` | Two-sided paired Wilcoxon signed-rank p-value, GCTR vs this method, on per-instance evaluation counts. |

### `Figure3_CalibrationAndUncertainty.csv`

Reliability curve of the Gaussian heads over ten nominal coverage levels.
`covered_count` is the number of standardized residuals covered at that
nominal level (cumulative in the level, not a disjoint bin membership).

### `Figure4_Generalization.csv`

Cross-size transfer for a model trained at the training size: mean
evaluations to target per size for random restarts, the concentration
heuristic, the GNN point baseline and GCTR, plus GCTR's speedup vs random.

### `Figure5_Ablation.csv`

Component ablation: evaluations to target, coverage ECE, and Spearman rho
(with p-value) per variant. Policy-level variants (trust region, heuristic
seed, adaptive budget, error calibrator) reuse the full trained model;
loss/feature variants are retrained.

### `Figure7_BudgetPolicy.csv`

Realized per-instance budget allocations on the test set: standardized
calibrated uncertainty `z`, seed count `K_seeds`, and evaluation cap
`T_budget_cap`.

### `EDFig2_ShotNoise.csv`

Finite-shot robustness: evaluations to target and exact expectation ratio for
Random / Heuristic / GCTR when the optimizer sees only S-shot estimates.

### `EDFig3_SeedStability.csv`

The query benchmark re-run across training/search seeds; one row per
(seed, method).
