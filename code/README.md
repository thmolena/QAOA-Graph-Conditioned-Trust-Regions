# `specops-gctr`

Installable reference implementation of graph-conditioned trust-region search
for QAOA. The distribution includes exact-statevector and finite-shot MaxCut
objectives, a graph-conditioned Gaussian model, a held-out error score,
query-counted search policies, baselines, plotting, tables, and the complete
manuscript reproduction driver.

## Install and verify

```bash
# Choose one install location:
python -m pip install .          # from code/
python -m pip install ./code     # from the repository root

# For development, from the repository root:
python -m pip install -e "./code[dev]"
python -m pytest code/tests -q
```

The package requires Python 3.9 or newer. CI spans every Python 3.9--3.13 release
and checks both the wheel and source distribution.

## Commands

```bash
gctr-optimize --family er --n 10 --budget 100 --json
gctr-reproduce --replot-only
gctr-reproduce --validate-only
gctr-reproduce --quick
gctr-reproduce
```

`gctr-optimize` exposes user-selected means, diagonal variances, prior angles,
trust-region radius, seed allocation, budget cap, stopping target, and shot
count. `gctr-reproduce` executes the released scientific protocol.

## Complete learned-policy API

```python
import numpy as np
from specops_gctr import Config, GCTRPolicyModel

p = 2
train_instances = [...]               # specops_gctr.Instance objects
target_angles = np.asarray(...)       # (n_train, 2 * p)
calibration_instances = [...]         # disjoint; at least two instances

policy = GCTRPolicyModel.fit(
    train_instances,
    target_angles,
    calibration_instances,
    config=Config(p_depth=p),
)
prediction = policy.predict(new_instance, budget=100)
result = policy.optimize(new_instance, budget=100)
policy.save("gctr-policy.pt")
policy = GCTRPolicyModel.load("gctr-policy.pt")
```

The training and calibration sets must be disjoint. Angle targets are needed
only for training graphs; they must have one `2 * p` vector per graph. The
calibration graphs fit the held-out warm-start error score and robust allocation
statistics. Load only trusted checkpoints, particularly with older PyTorch
versions whose deserializer predates restricted weight loading.

No pretrained checkpoint ships with the distribution. The optional
`target_fraction` argument uses a fraction of exact MaxCut; it is distinct from
the manuscript's fraction of an offline per-instance reference value.

`predict(new_instance)` uses an exact-simulation `Instance`. For deployment
without exact MaxCut or a `2**n` cost diagonal, use graph-only prediction and
the callback policy:

```python
pred = policy.predict_graph(networkx_graph, budget=100)
result = gctr_callback_policy(
    evaluator=backend.objective,
    budget=100,
    mu=pred.mean,
    sigma_diag=pred.variance,
    heuristic_angles=policy.concentration,
    n_gaussian_seeds=pred.allocation["n_gaussian_seeds"],
    budget_cap=pred.allocation["T"],
)
```

Graph-only prediction removes the exponential simulator dependency but still
uses a dense Laplacian eigendecomposition and dense adjacency tensor (roughly
cubic time and quadratic memory). It is not an arbitrary-scale sparse graph
backend.

Fitting is more restrictive than prediction: `GCTRPolicyModel.fit` currently
requires exact-simulation `Instance` objects for its calibration graphs and
does not expose an external calibration-error or objective-label callback.
Users with large or hardware-labelled graphs must supply a trusted fitted model
or extend this interface; the release does not imply a scalable training path.

## Low-level policy API

```python
import numpy as np
from specops_gctr import gctr_policy, make_instance

instance = make_instance("rr", n=8, seed=0)
result = gctr_policy(
    instance,
    budget=60,
    mu=np.array([0.4, 0.6, 0.8, 0.3]),
    sigma_diag=np.full(4, 0.05),
    target=None,  # full-budget search; or provide an absolute objective target
)
```

For a QPU, remote runtime, emulator, or external scalar service, use the same
policy geometry through a callback:

```python
from specops_gctr import gctr_callback_policy

result = gctr_callback_policy(
    evaluator=lambda theta: backend.objective(theta),
    budget=60,
    mu=np.array([0.4, 0.6, 0.8, 0.3]),
    sigma_diag=np.full(4, 0.05),
)
```

The callback count is exact. Results include `evaluations_used` and
`reached_target`; fair cross-method scoring assigns a shared cap to a target
miss. Physical shots, mitigation, compilation, and backend uncertainty are
deliberately left to the integration layer.

Best-so-far selection preserves the best evaluated seed. The corresponding
seed-safety statement requires an effective budget large enough to evaluate the
mandatory seed set; it is not a guarantee relative to a separately refined
baseline.

`gctr-reproduce --replot-only` consumes committed manuscript data and is
therefore a checkout workflow unless `--source-data-dir` is supplied. The wheel
does not duplicate those files; its simulator, estimator, callback adapter, and
new experiment paths work independently of the repository.

The committed experiment uses schema 2: raw evaluations, shared-cap scores,
attainment flags, primary and leave-one-family-out instance rows, implementation
fingerprints, and artifact hashes are recorded. `--validate-only` checks the
manifest without rerunning the experiment.

## Distribution checks

```bash
python -m build .
python -m twine check dist/*
python -m pip install dist/*.whl
gctr-optimize --family er --n 4 --budget 4 --json
```

See the repository-level README for the claim ledger, full reproduction paths,
and scientific limitations. The released exact-statevector evidence is limited
to `n <= 16`, and the backend itself scales exponentially with qubit count.

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

MIT licensed.
