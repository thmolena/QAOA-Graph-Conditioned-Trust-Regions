# specops-gctr

Reference implementation for **Graph-Conditioned Trust Regions for Query-Efficient QAOA**.
Exact-statevector QAOA on MaxCut, a trained graph-conditioned Gaussian surrogate over the
angles with a held-out error calibrator, and the query-efficiency / calibration /
cross-size / leave-one-family-out / ablation / seed-stability / shot-noise studies —
everything needed to regenerate the manuscript's data figures, tables, and CSVs.

## Install
```bash
pip install .            # or:  pip install -e .   (editable)
pip install "git+https://github.com/thmolena/QAOA-Graph-Conditioned-Trust-Regions.git#subdirectory=code"
pip install -e ".[test]" # with test deps
```

## Reproduce
```bash
gctr-reproduce                # full manuscript run (tens of minutes on a laptop)
gctr-reproduce --quick        # smoke config (~1 min)
gctr-reproduce --replot-only  # re-render from committed source data (seconds)
```

Algorithmic outputs are deterministic given the seed (`--seed`, default
`20260424`) and a fixed software environment; the environment used for the
committed run is recorded in `../manuscript/source_data/meta.json`. Learned-model numbers
can shift slightly across torch/BLAS versions, and wall-clock `runtime_ms`
columns are machine-dependent by nature. Model-free baselines reproduce
bit-for-bit everywhere.

## Test
```bash
pytest -q
```
19 tests: trust-region feasibility, exact query counting (seed evaluations
included), budget caps, the QAOA objective against an independent brute-force
computation, ECE sanity on synthetic residuals, budget-rule monotonicity, the
TQA schedule, dataset determinism, and the replot path over committed
reference data.

## Layout
- `specops_gctr/` — `qaoa` (simulator + finite-shot estimator), `graphs`,
  `targets`, `model` (GIN + error calibrator), `losses`, `optimization`
  (query-counted pattern search + trust-region retraction), `baselines`,
  `calibration`, `pipeline` (all studies + budget rule), `plots`, `tables`,
  and the `reproduce` driver.
- `tests/` — pytest suite; the committed reference data live in
  `../manuscript/source_data/`.

MIT licensed. Every number in the paper is produced by this package; nothing
is hand-entered.
