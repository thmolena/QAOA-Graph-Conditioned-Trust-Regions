"""Replay immutable GCTR evidence or launch a new seeded end-to-end rerun."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources as resources
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PACKAGE = "gctr_repro"
LEGACY_SEED = 20260424
PORTFOLIO_STUDIES = (
    (
        "configs/portfolio_development.json",
        "portfolio_results/development_v2_tqa_family_gate",
    ),
    (
        "configs/portfolio_heterogeneous_development.json",
        "portfolio_results/heterogeneous_development_v2_tqa_family_gate",
    ),
    (
        "configs/portfolio_heterogeneous_confirmatory.json",
        "portfolio_results/heterogeneous_confirmatory_v1",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict:
    payload = resources.files(PACKAGE).joinpath(
        "evidence_manifest.json"
    ).read_text(encoding="utf-8")
    manifest = json.loads(payload)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported GCTR evidence manifest")
    return manifest


def _verify_files(bundle: Path, manifest: dict) -> dict:
    expected = manifest.get("files", {})
    if not expected:
        raise RuntimeError("GCTR evidence manifest has no files")
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    expected_paths = set(expected)
    errors = []
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        added = sorted(actual_paths - expected_paths)
        if missing:
            errors.append("missing files: " + ", ".join(missing))
        if added:
            errors.append("unmanifested files: " + ", ".join(added))
    for relative, record in sorted(expected.items()):
        path = bundle / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size != record["size"]:
            errors.append(
                f"size mismatch for {relative}: {size} != {record['size']}"
            )
            continue
        digest = _sha256(path)
        if digest != record["sha256"]:
            errors.append(f"SHA-256 mismatch for {relative}")
    if errors:
        raise RuntimeError("GCTR evidence replay failed:\n- " + "\n- ".join(errors))
    return {
        "files": len(expected),
        "bytes": sum(item["size"] for item in expected.values()),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _validate_domain_manifests(bundle: Path) -> dict:
    from specops_gctr.portfolio_experiment import (
        load_config,
        validate_portfolio_manifest,
    )
    from specops_gctr.reproduce import validate_manifest

    legacy = validate_manifest(bundle / "manuscript")
    portfolio = {}
    for config_relative, result_relative in PORTFOLIO_STUDIES:
        config = load_config(bundle / config_relative)
        value = validate_portfolio_manifest(bundle / result_relative, config)
        portfolio[result_relative] = value["decision_sha256"]
    return {
        "legacy_schema": legacy["schema_version"],
        "portfolio_decisions": portfolio,
    }


def replay(output: Path | None = None) -> dict:
    manifest = _load_manifest()
    bundle_ref = resources.files(PACKAGE).joinpath("bundle")
    with resources.as_file(bundle_ref) as bundle:
        file_report = _verify_files(bundle, manifest)
        domain_report = _validate_domain_manifests(bundle)
        if output is not None:
            destination = output.expanduser().resolve()
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(
                    f"replay output must be absent or empty: {destination}"
                )
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                bundle,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            _verify_files(destination, manifest)
    return {"mode": "byte-hash-replay", **file_report, **domain_report}


def _command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def _run_commands(
    commands: Iterable[list[str]], *, cwd: Path, dry_run: bool
) -> None:
    for command in commands:
        print("+", " ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=cwd, check=True)


def _prepare_full_tree(output: Path) -> None:
    bundle_ref = resources.files(PACKAGE).joinpath("bundle")
    with resources.as_file(bundle_ref) as bundle:
        shutil.copytree(bundle / "configs", output / "configs")
        (output / "manuscript").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            bundle / "manuscript/generate_portfolio_artifacts.py",
            output / "manuscript/generate_portfolio_artifacts.py",
        )


def full_rerun(
    output: Path,
    *,
    seed: int,
    quick: bool,
    dry_run: bool,
) -> dict:
    destination = output.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"full-rerun output must be absent or empty: {destination}")
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        _prepare_full_tree(destination)

    legacy_args: list[object] = [
        "--manuscript-dir",
        destination / "manuscript",
        "--seed",
        seed,
    ]
    if quick:
        legacy_args.append("--quick")
    commands = [_command("specops_gctr.reproduce", *legacy_args)]
    if not quick:
        for config_relative, result_relative in PORTFOLIO_STUDIES:
            commands.append(
                _command(
                    "specops_gctr.portfolio_experiment",
                    "--config",
                    destination / config_relative,
                    "--output-dir",
                    destination / result_relative,
                )
            )
        commands.append(
            [sys.executable, str(
                destination / "manuscript/generate_portfolio_artifacts.py"
            )]
        )
    _run_commands(commands, cwd=destination, dry_run=dry_run)
    report = {
        "mode": "full-seeded-rerun",
        "seed": seed,
        "quick": quick,
        "dry_run": dry_run,
        "commands": commands,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bitwise_identity_claimed": False,
        "numerical_drift_note": (
            "PyTorch, BLAS, compiler, and platform differences may change "
            "floating-point results despite fixed seeds."
        ),
    }
    if not dry_run:
        (destination / "rerun_metadata.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gctr-reproduce-all",
        description=(
            "Verify/copy committed GCTR bytes or launch every seeded experiment."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    replay_parser = subparsers.add_parser(
        "replay", help="hash-verify committed artifacts without recomputation"
    )
    replay_parser.add_argument(
        "--output", type=Path, default=None,
        help="optional empty directory receiving an exact byte copy",
    )
    full_parser = subparsers.add_parser(
        "full", help="run a new seeded execution (not expected to be bitwise portable)"
    )
    full_parser.add_argument("--output", type=Path, required=True)
    full_parser.add_argument("--seed", type=int, default=LEGACY_SEED)
    full_parser.add_argument(
        "--quick", action="store_true",
        help="run only the reduced legacy smoke configuration",
    )
    full_parser.add_argument(
        "--dry-run", action="store_true", help="print the execution plan"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "replay":
        report = replay(args.output)
    else:
        report = full_rerun(
            args.output, seed=args.seed, quick=args.quick, dry_run=args.dry_run
        )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
