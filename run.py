"""Repository dispatcher for the two supported command-line workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent


USAGE = """usage: python run.py {optimize,reproduce,portfolio,release} [arguments]

commands:
  optimize   run one configurable GCTR optimization
  reproduce  run, replot, or validate the manuscript experiment
  portfolio  run the target-free, fixed-budget optimizer portfolio
  release    regenerate and validate the complete publication package
"""


def _checked(command: list[str], *, cwd: Path = ROOT) -> None:
    """Run one release command and stop immediately on drift or failure."""
    print("+", " ".join(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _validate_bundle_sync() -> None:
    """Require every bundled study/manuscript byte to match the repository."""
    bundle = ROOT / "code/src/gctr_repro/bundle"
    mismatches: list[str] = []
    checked = 0
    for subtree in ("configs", "manuscript", "portfolio_results"):
        for packaged in sorted((bundle / subtree).rglob("*")):
            if (
                not packaged.is_file()
                or "__pycache__" in packaged.parts
                or packaged.suffix in {".pyc", ".pyo"}
            ):
                continue
            relative = packaged.relative_to(bundle)
            canonical = ROOT / relative
            checked += 1
            if not canonical.is_file() or canonical.read_bytes() != packaged.read_bytes():
                mismatches.append(relative.as_posix())
    if mismatches:
        raise RuntimeError(
            "bundled evidence/source drift: " + ", ".join(mismatches)
        )
    print(f"[bundle-sync] {checked} evidence/source files are byte-identical")


def _compile_and_test_archive() -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic is required for the release build")
    manuscript = ROOT / "manuscript"
    _checked(
        [tectonic, "--keep-logs", "--keep-intermediates", "main.tex"],
        cwd=manuscript,
    )
    _checked(
        [sys.executable, "manuscript/validate_manuscript.py", "--skip-archive"]
    )
    _checked([sys.executable, "manuscript/build_arxiv_archive.py"])
    archive = manuscript / "arxiv-source-gctr.zip"
    with tempfile.TemporaryDirectory(prefix="gctr-source-build-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
        _checked([tectonic, "--keep-logs", "main.tex"], cwd=extracted)
        pdf = extracted / "main.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 100_000:
            raise RuntimeError("extracted GCTR source archive did not build a valid PDF")
    _checked([sys.executable, "manuscript/validate_manuscript.py"])


def _release() -> None:
    """Regenerate all supported displays and enforce the publication gate."""
    _checked([sys.executable, "manuscript/generate_portfolio_artifacts.py"])
    _checked([sys.executable, "code/tools/check_source_sync.py"])
    _validate_bundle_sync()
    package_environment = dict(os.environ)
    package_environment["PYTHONPATH"] = str(ROOT / "code/src")
    package_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    package_environment["PYTHONUTF8"] = "1"
    print("+ gctr-reproduce-all replay", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from gctr_repro.cli import replay; import json; "
            "print(json.dumps(replay(), sort_keys=True))",
        ],
        cwd=ROOT,
        env=package_environment,
        check=True,
    )
    _compile_and_test_archive()
    print("[release] GCTR evidence, manuscript, package mirror, and archive passed")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0

    command, forwarded = args[0], args[1:]
    if command == "optimize":
        from specops_gctr.cli import main as optimize_main
        return optimize_main(forwarded)
    if command == "reproduce":
        from specops_gctr.reproduce import main as reproduce_main
        return reproduce_main(forwarded)
    if command == "portfolio":
        from specops_gctr.portfolio_experiment import main as portfolio_main
        return portfolio_main(forwarded)
    if command == "release":
        if forwarded:
            print("release does not accept additional arguments", file=sys.stderr)
            return 2
        _release()
        return 0

    print(f"unknown command: {command!r}\n\n{USAGE}", file=sys.stderr,
          end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
