# SpecOps GCTR self-contained code package

This directory is an independently buildable `src`-layout distribution. It
contains a byte-identical mirror of the canonical root `specops_gctr/` package,
the committed machine-readable evidence and generated artifacts, the frozen
portfolio configurations, the original reproduction scripts, and a single
command that makes the distinction between artifact replay and a new
simulation explicit.

## Fresh-clone Mac CPU install

Python 3.9 through 3.13 is supported. On a fresh clone:

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[test,release]"
```

For the exact resolver state recorded in `uv.lock`, install `uv` and use:

```bash
uv sync --locked --extra test --extra release
source .venv/bin/activate
```

The lock is universal across the declared Python range and records
platform-specific wheels and markers. Availability of PyTorch wheels still
depends on the selected Python version and CPU architecture.

## Verify the immutable evidence and source mirror

The replay command hashes every bundled byte, validates the legacy manuscript
manifest, and validates all three committed optimizer-portfolio manifests. It
does not run training, optimization, or plotting:

```bash
gctr-reproduce-all replay
python tools/check_source_sync.py
python -m pytest
```

To materialize an exact byte-for-byte copy of the committed evidence:

```bash
gctr-reproduce-all replay --output replayed-evidence
```

`replayed-evidence/` contains the same repository-relative paths and SHA-256
payloads recorded in `src/gctr_repro/evidence_manifest.json`.

## Run every seeded experiment again

The full mode runs the legacy schema-2 training/evaluation pipeline with seed
`20260424`, then independently reruns the regular development, heterogeneous
development, and heterogeneous confirmatory portfolio configurations using the
seeds frozen in their JSON files. Finally, it regenerates the portfolio figures
and tables:

```bash
gctr-reproduce-all full --output full-rerun --seed 20260424
```

This is intentionally expensive. Inspect the command plan without executing:

```bash
gctr-reproduce-all full --output full-rerun --dry-run
```

A reduced smoke run of the legacy pipeline is available, but it is not a
substitute for the three portfolio studies:

```bash
gctr-reproduce-all full --output quick-rerun --quick
```

Replay and rerun answer different questions:

- `replay` proves that the released CSV, JSON, JSONL, checkpoint, figure, table,
  and protocol bytes are unchanged.
- `full` produces a new seeded execution. Seeds and deterministic settings are
  recorded, but floating-point reductions, BLAS, PyTorch, compiler, and
  platform differences can prevent cross-platform bitwise identity.

The design-only `prospective_risk_control_v2.design.json` is included and
hashed. It is intentionally not executed because its own status is
`design_only_not_registered_not_run`.

## Build and wheel-install checks

```bash
python -m build .
python -m twine check dist/*
python -m venv /tmp/gctr-wheel-venv
source /tmp/gctr-wheel-venv/bin/activate
python -m pip install dist/specops_gctr-3.2.0-py3-none-any.whl
gctr-reproduce-all replay
gctr-optimize --family er --n 4 --budget 4 --json
```

The canonical implementation remains the repository-root
`specops_gctr/` directory. `tools/check_source_sync.py` rejects missing, added,
or byte-different files and verifies `source_manifest.json`, preventing the
packaging mirror from drifting silently.
