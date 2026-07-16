# Graph-Conditioned Trust Regions for Query-Efficient QAOA

[![Package CI](https://github.com/thmolena/QAOA-Graph-Conditioned-Trust-Regions/actions/workflows/ci.yml/badge.svg)](https://github.com/thmolena/QAOA-Graph-Conditioned-Trust-Regions/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

QAOA objective values are statistical queries: every optimizer evaluation costs
repeated circuit preparation and measurement. This project learns a complete
search policy from graph structure—a center, anisotropic step geometry,
Mahalanobis trust region, seed set, and validation-fit per-instance budget—rather
than learning only one warm-start vector.

The released exact-statevector study is deliberately bounded. At depth `p=2`
on 48 held-out `n=14` MaxCut instances, GCTR reaches the shared target on
`48/48` graphs with capped target cost `15.0 +/- 7.8` objective calls and mean
exact expectation ratio `0.858`. The fixed concentration heuristic also reaches
`48/48` and is cheaper (`12.33 +/- 6.45`, paired Wilcoxon
`p = 2.6488e-4`). TQA reaches `48/48` at `18.48 +/- 13.48` calls and is not
distinguishable from GCTR on this comparison (`p = 0.0683`). Random restarts,
k-NN, and the GNN point initializer reach `37/48`, `42/48`, and `20/48`, with
capped costs `203.9 +/- 152.9`, `52.94 +/- 131.93`, and
`264.42 +/- 164.02`, respectively. A miss is scored at the common 400-call cap
while its raw work and attainment flag remain recorded. The evidence supports
a simulated low-depth case study, not hardware or asymptotic advantage.
The heuristic is cheap only online: its fixed angles are the coordinate-wise
median of 240 expensive training targets.

- [Manuscript](manuscript/main.pdf) · [LaTeX source](manuscript/main.tex)
- [Project page](https://thmolena.github.io/QAOA-Graph-Conditioned-Trust-Regions/)
- [Python package](specops_gctr/) · [Source data](manuscript/source_data/)
- Preprint: [arXiv:2604.24803](https://arxiv.org/abs/2604.24803)

> **Revision status.** The repository manuscript, package, and schema-2
> numerical artifacts are newer than arXiv v1. Treat the local PDF as the
> working revision until the updated preprint is uploaded.

## Claims and verification paths

| Supported statement | Evidence | Reproduction path | Boundary |
| --- | --- | --- | --- |
| GCTR has 92.6% lower mean capped cost than random restarts and higher target attainment | GCTR `48/48`, `15.0 +/- 7.8`; random `37/48`, `203.9 +/- 152.9`; paired `p = 2.86e-9` | `gctr-reproduce --replot-only`; `Figure2_QueryEfficiency.csv`; `QueryEfficiency_InstanceLevel.csv` | The target is a privileged offline reference and the ratio of capped costs is not a hardware speedup |
| The fixed heuristic is the strongest in-distribution online-cost baseline | Heuristic `48/48`, `12.33 +/- 6.45`; GCTR `48/48`, `15.0 +/- 7.8`; paired `p = 2.6488e-4` | `Figure2_QueryEfficiency.csv`; `QueryEfficiency_InstanceLevel.csv` | Its angles use the 240 supervised training targets; GCTR does not improve its online query cost in this test |
| TQA and GCTR both attain every in-distribution target | TQA `48/48`, `18.48 +/- 13.48`; paired `p = 0.0683` | `Figure2_QueryEfficiency.csv`; `QueryEfficiency_InstanceLevel.csv` | This test does not establish a cost difference |
| Gaussian angle uncertainty has marginal coverage diagnostics; a separate error score drives allocation | Angle-head ECE `0.101`; difficulty-score Spearman `rho = 0.671`, `p = 1.76e-7` | `Figure3_CalibrationAndUncertainty.csv`; `Figure7_BudgetPolicy.csv` | ECE and rank correlation measure different objects; neither proves calibrated stopping probabilities |
| LOFO shift exposes mixed method ordering | GCTR `39/48`, cost `89.3`; heuristic `40/48`, `86.5`, `p = 0.832`; point `38/48`, `125.0`, `p = 0.00509`; two-seed/full-cap control `41/48`, `73.75`, `p = 0.0355` | `LOFO_InstanceLevel.csv`; `results.json` | Four fitted family clusters and a composite control make the paired tests exploratory, not causal evidence for allocation |
| The ranking penalty is not supported by this run | Full ECE/rank correlation `0.101/0.671`; without ranking penalty `0.026/0.854` | `Figure5_Ablation.csv`; `results.json` | One component-removal run is diagnostic, not a causal estimate or a new confirmatory test |
| Returned exact quality is protected by best-so-far seed selection when the budget admits the mandatory seeds | Monotone implementation and `test_seeded_policy_never_worse_than_best_seed` | `specops_gctr/optimization.py`; focused test | The guarantee is conditional on enough effective budget to evaluate the required seed set and compares with evaluated seed values, not a separately refined baseline |
| Early truncation cannot improve shared-cap cost on a fixed trajectory | Pointwise target-hit/miss argument in the early-cap monotonicity proposition | manuscript Sec. V; `score_semantics` in `meta.json` | The two-seed/full-cap control also changes seeds, so its empirical difference cannot be assigned to the cap |
| If a policy uses fewer fixed evaluation locations, a sufficient uniform-shot upper bound decreases | Concentration bounds plus finite-shot stress tests | manuscript Sec. V and Extended Data Fig. 2 | GCTR does not use fewer locations than the heuristic/control here; this is not an end-to-end hardware-cost identity |

### Cross-size transfer with attainment accounting

One model trained at `n=14` is evaluated from `n=8` through `n=16`. The final
column is the arithmetic ratio of random-restart to GCTR mean capped cost; it is
not a runtime or hardware speedup. The online-cheap fixed heuristic is lower at
four of five sizes; GCTR is slightly lower only at `n=10`, with identical
attainment at every size.

| Vertices | GCTR cost / hits | Heuristic cost / hits | Random cost / hits | Random/GCTR |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 11.6 / 48 | 10.9 / 48 | 156.9 / 40 | 13.47x |
| 10 | 20.1 / 47 | 21.8 / 47 | 170.2 / 39 | 8.47x |
| 12 | 13.4 / 48 | 12.1 / 48 | 186.2 / 37 | 13.95x |
| 14 | 15.4 / 48 | 12.5 / 48 | 198.0 / 37 | 12.82x |
| 16 | 15.2 / 48 | 10.5 / 48 | 158.2 / 39 | 10.40x |

## Install

Install the canonical package from a checkout:

```bash
python -m pip install .
```

Or install directly from the public Git repository:

```bash
python -m pip install \
  "specops-gctr @ git+https://github.com/thmolena/QAOA-Graph-Conditioned-Trust-Regions.git"
```

For development and release checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

The distribution name is `specops-gctr` and the import name is
`specops_gctr`. Bare `pip install specops-gctr` will work only after version
3.2.0 is uploaded to a Python package index; this repository does not claim
that the index upload has already happened.

The installed console commands remain the primary interface. From a checkout,
the root dispatcher offers the same two entry points without path-specific
shell wrappers:

```bash
python run.py optimize --help
python run.py reproduce --validate-only
```

## Fit and use the complete learned policy

`GCTRPolicyModel` is the high-level train/predict/optimize interface. It learns
the graph-conditioned Gaussian, fits the held-out error head, retains the
training-set concentration prior, and can be saved and loaded:

```python
import numpy as np
from specops_gctr import Config, GCTRPolicyModel

# Each item is a specops_gctr.Instance. Generate target angles offline using
# the optimizer appropriate to your study.
p = 2
train_instances = [...]
target_angles = np.asarray(...)       # shape: (len(train_instances), 2 * p)
calibration_instances = [...]         # disjoint from training; at least 2

policy = GCTRPolicyModel.fit(
    train_instances,
    target_angles,
    calibration_instances,
    config=Config(p_depth=p),
)
prediction = policy.predict(new_instance, budget=100)
result = policy.optimize(new_instance, budget=100)
policy.save("gctr-policy.pt")
restored = GCTRPolicyModel.load("gctr-policy.pt")
```

The calibration set is intentionally disjoint: it fits the error score and its
allocation statistics without reusing angle-training graphs. Target angles are
required only for training instances. Load only checkpoints you trust,
especially with older PyTorch versions whose deserializer predates restricted
weight loading.

No pretrained checkpoint is included. Users fit the policy on their own labelled
graph distribution or load a trusted checkpoint they created. The optional
`target_fraction` API multiplies an instance's exact MaxCut value; it is not the
paper's `0.98 * V_i` stopping rule, whose `V_i` is an offline reference value.

`Instance.from_networkx` is intentionally an exact-simulation constructor: it
computes exact MaxCut and a `2**n` cost diagonal, so it is suitable only for
small validation problems. For a larger deployment graph, predict the policy
without building either exponential object and pass it to the callback adapter:

```python
import networkx as nx
from specops_gctr import gctr_callback_policy

graph = nx.read_edgelist("production-graph.edgelist")
pred = restored.predict_graph(graph, budget=100)
result = gctr_callback_policy(
    measured_expectation,
    budget=100,
    mu=pred.mean,
    sigma_diag=pred.variance,
    heuristic_angles=restored.concentration,
    n_gaussian_seeds=pred.allocation["n_gaussian_seeds"],
    budget_cap=pred.allocation["T"],
)
```

This graph-only path avoids exact simulation, but the current dense Laplacian
eigendecomposition and adjacency tensor still scale as roughly cubic time and
quadratic memory in the vertex count. Sparse features and a sparse GNN backend
are required before claiming arbitrary industry-scale graph support.

## Use a precomputed Gaussian

The public API accepts a graph instance and a user-supplied Gaussian policy, so
the trust-region geometry, budget, shot count, and seed design are all tunable:

```python
import numpy as np
from specops_gctr import gctr_policy, make_instance

instance = make_instance("er", n=10, seed=7)
result = gctr_policy(
    instance,
    budget=100,
    mu=np.array([0.40, 0.60, 0.80, 0.30]),
    sigma_diag=np.array([0.05, 0.03, 0.08, 0.04]),
    radius_scale=2.0,
    target=None,  # use the full budget; or supply an absolute objective target
    heuristic_angles=np.array([0.35, 0.62, 0.75, 0.28]),
    n_gaussian_seeds=2,
    shots=None,
)

print(result["value"], result["evaluations"], result["params"])
```

The corresponding single-instance command is useful for sweeps and workflow
systems:

```bash
gctr-optimize --family er --n 10 --seed 7 \
  --mean 0.40 0.60 0.80 0.30 \
  --variance 0.05 0.03 0.08 0.04 \
  --heuristic 0.35 0.62 0.75 0.28 \
  --budget 100 --gaussian-seeds 2 --json
```

`gctr-optimize --help` lists the trust-region, budget, target, and finite-shot
controls. The bundled simulator scales exponentially with qubit count and is a
research reference, not a large-device execution backend.

### Connect a hardware or service backend

`gctr_callback_policy` decouples the search policy from the bundled simulator.
The callback can submit a circuit to a QPU/runtime, invoke an emulator, or query
an industrial objective service; every callback invocation is counted exactly:

```python
import numpy as np
from specops_gctr import gctr_callback_policy

def measured_expectation(theta):
    # Replace with a backend call returning one scalar objective estimate.
    return runtime.evaluate_qaoa(theta, shots=2_000)

result = gctr_callback_policy(
    measured_expectation,
    budget=60,
    mu=np.array([0.40, 0.60, 0.80, 0.30]),
    sigma_diag=np.array([0.05, 0.03, 0.08, 0.04]),
    radius_scale=2.0,
)
```

One callback call is one objective estimate, not one physical shot. Backend
authentication, compilation, mitigation, queueing, and shot allocation remain
the caller's responsibility; this adapter is an integration boundary, not new
hardware evidence. Results expose `evaluations_used` and `reached_target`.
When comparing methods under different early caps, score a miss at the common
benchmark cap; the reproduction pipeline does this explicitly.

## Reproduce the paper

The artifact exposes four verification levels:

```bash
gctr-reproduce --replot-only  # checkout only: committed CSV/JSON -> artifacts
gctr-reproduce --validate-only # checkout only: verify manifest and hashes
gctr-reproduce --quick        # reduced end-to-end smoke experiment
gctr-reproduce                # canonical training and evaluation run
```

The wheel contains the complete executable algorithm but does not duplicate the
repository's committed manuscript source data. Run `--replot-only` from a
checkout (or pass `--source-data-dir`); clean wheel installs can run
`gctr-optimize`, the Python APIs, and new `--quick` or full experiments.

The committed source data use schema 2. They record the generator and package
versions, implementation fingerprint, raw evaluations, capped benchmark cost,
target attainment, primary and LOFO instance rows, and hashes of generated
artifacts. `gctr-reproduce --validate-only` checks this provenance chain. The
default seed is `20260424`; trained-model values can vary slightly across
PyTorch, BLAS, and platform builds. The recorded environment is stored in
[`manuscript/source_data/meta.json`](manuscript/source_data/meta.json), and
wall-clock columns are machine-dependent.

## Distribution checks

```bash
python -m build .
python -m twine check dist/*
python -m pip install dist/*.whl
gctr-optimize --family er --n 4 --budget 4 --json
```

The wheel contains the executable package and console commands; committed
manuscript evidence stays in the repository and is supplied explicitly to
replot or validation commands.

## Module contract

| Module | Responsibility |
| --- | --- |
| `qaoa` | Exact and finite-shot QAOA objectives |
| `graphs` | Instances, exact MaxCut, spectral encodings |
| `model`, `losses`, `estimator` | GIN Gaussian, training objectives, serializable high-level policy |
| `calibration` | Reliability and uncertainty diagnostics |
| `optimization`, `baselines` | Query-counted search, callback adapter, comparisons |
| `pipeline` | Data, training, calibration, allocation, evaluations |
| `plots`, `tables` | Source data to publication artifacts |
| `cli`, `reproduce` | User command and full artifact command |

## Repository map

```text
specops_gctr/            canonical package, estimator, and two console commands
tests/                    mathematical-contract, API, CLI, and reproduction tests
run.py                    root dispatcher for optimize and reproduce commands
pyproject.toml            single package and build definition
manuscript/main.tex      article source
manuscript/main.pdf      compiled article
manuscript/source_data/  aggregate and per-instance CSV/JSON evidence
manuscript/figures/      generated PDF/PNG figures
manuscript/tables/       generated LaTeX tables
index.html               static project page
.github/workflows/       package CI and GitHub Pages deployment
```

The source-data directory is the single source of truth for published
aggregates. Data figures and numerical tables are generated from it; the three
conceptual schematics are deterministic package outputs.

## Scope and extension points

- The release studies unweighted MaxCut, exact statevectors, `p=2`, and
  `n <= 16`; no device, compilation, queueing, or readout-noise result is claimed.
- The revision inspected the released test results while correcting the
  protocol and claims. A future confirmatory claim requires a newly generated,
  prospectively untouched audit set.
- `GCTRPolicyModel` provides the complete fit/predict/optimize/save/load path;
  `gctr_policy` and `gctr_callback_policy` expose lower-level integration points.
- Radius, step size, seed count, budget cap, finite shots, and the trust-region
  switch are tunable. New graph distributions require new training targets and
  a disjoint calibration split.
- The fixed-angle prior is a first-class online baseline derived from supervised
  training targets. A new study should account for those offline labels and not
  omit the prior merely because random-restart comparisons are more favorable.
- The released two-seed/full-cap control changes Gaussian seeding and the early
  cap together. In LOFO it records `41/48` attainment and cost `73.75`, versus
  GCTR's `39/48` and `89.3`; the comparison does not isolate either lever.
- The finite-shot results are recorded stress tests at two shot counts, not a
  repeated-noise or equal-total-shot benchmark.

## Citation and license

```bibtex
@article{huynh2026gctr,
  title   = {Query-Efficient Quantum Approximate Optimization via
             Graph-Conditioned Trust Regions},
  author  = {Huynh, Molena},
  journal = {arXiv preprint arXiv:2604.24803},
  year    = {2026}
}
```

MIT licensed. Contributions should include a focused test and preserve the
claim boundaries documented above.
