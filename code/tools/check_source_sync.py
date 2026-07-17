#!/usr/bin/env python3
"""Reject drift between root specops_gctr/ and code/src/specops_gctr/."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE = "specops_gctr"


def _files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body(canonical: Path) -> dict:
    return {
        "schema_version": 1,
        "canonical": PACKAGE,
        "files": {
            relative: {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for relative, path in sorted(_files(canonical).items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    canonical = repository / PACKAGE
    mirror = repository / "code/src" / PACKAGE
    manifest_path = repository / "code/source_manifest.json"
    expected = _body(canonical)
    if args.write_manifest:
        manifest_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not manifest_path.is_file():
        raise SystemExit(f"missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest != expected:
        errors.append("source_manifest.json does not match the canonical package")
    canonical_files = _files(canonical)
    mirror_files = _files(mirror)
    if set(canonical_files) != set(mirror_files):
        errors.append(
            "file set differs: missing="
            + repr(sorted(set(canonical_files) - set(mirror_files)))
            + " added="
            + repr(sorted(set(mirror_files) - set(canonical_files)))
        )
    for relative in sorted(set(canonical_files) & set(mirror_files)):
        if canonical_files[relative].read_bytes() != mirror_files[relative].read_bytes():
            errors.append(f"byte mismatch: {relative}")
    if errors:
        raise SystemExit("source sync failed:\n- " + "\n- ".join(errors))
    print(
        f"[source-sync] {len(canonical_files)} files are byte-identical; "
        f"manifest verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
