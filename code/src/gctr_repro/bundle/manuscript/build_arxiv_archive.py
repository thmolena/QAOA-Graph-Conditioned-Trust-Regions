#!/usr/bin/env python3
"""Build the deterministic 23-member GCTR PRX/arXiv source archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import validate_manuscript


MANUSCRIPT = Path(__file__).resolve().parent
OUTPUT = MANUSCRIPT / "arxiv-source-gctr.zip"
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_inputs() -> list[Path]:
    source = validate_manuscript.without_comments(
        (MANUSCRIPT / "main.tex").read_text()
    )
    sources = validate_manuscript.archive_sources(source)
    paths = [sources[name] for name in sorted(sources)]
    missing = [
        str(path.relative_to(MANUSCRIPT)) for path in paths if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"archive inputs are missing: {missing}")
    return paths


def write_archive(paths: list[Path]) -> None:
    temporary = OUTPUT.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in paths:
            relative = path.relative_to(MANUSCRIPT).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary.replace(OUTPUT)


def verify_archive(paths: list[Path]) -> None:
    expected = sorted(path.relative_to(MANUSCRIPT).as_posix() for path in paths)
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if names != expected:
            raise RuntimeError(f"archive member mismatch: {names}")
        for path, member in zip(paths, names):
            if archive.read(member) != path.read_bytes():
                raise RuntimeError(f"archive byte mismatch: {member}")


def main() -> int:
    preflight = validate_manuscript.validate(require_archive=False)
    print("[arxiv] preflight validation passed")
    print(json.dumps(preflight, indent=2, sort_keys=True))
    paths = archive_inputs()
    write_archive(paths)
    verify_archive(paths)
    final = validate_manuscript.validate(require_archive=True)
    print(f"[arxiv] wrote {OUTPUT}")
    print(f"  members: {len(paths)}")
    print(f"  sha256: {sha256(OUTPUT)}")
    print("[arxiv] post-build validation passed")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
