#!/usr/bin/env python3
# Final deterministic arXiv source builder.
"""Build and verify a deterministic, dependency-closed arXiv source archive."""

from __future__ import annotations

import hashlib
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


def archive_inputs() -> dict[str, Path]:
    source = validate_manuscript.without_comments(
        (MANUSCRIPT / "main.tex").read_text()
    )
    paths = validate_manuscript.archive_sources(source)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"archive inputs are missing: {missing}")
    return dict(sorted(paths.items()))


def write_archive(paths: dict[str, Path]) -> None:
    temporary = OUTPUT.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative, path in paths.items():
            info = zipfile.ZipInfo(relative, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary.replace(OUTPUT)


def verify_archive(paths: dict[str, Path]) -> None:
    expected = list(paths)
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if names != expected:
            raise RuntimeError(
                f"archive member mismatch: expected {expected}, observed {names}"
            )
        for relative, path in paths.items():
            member = archive.getinfo(relative)
            if member.date_time != FIXED_TIMESTAMP:
                raise RuntimeError(f"archive timestamp differs: {relative}")
            if member.external_attr >> 16 != 0o100644:
                raise RuntimeError(f"archive permissions differ: {relative}")
            if archive.read(relative) != path.read_bytes():
                raise RuntimeError(f"archive byte mismatch: {relative}")


def main() -> int:
    # The archive cannot be a precondition for building itself.
    validate_manuscript.validate(require_archive=False)
    paths = archive_inputs()
    write_archive(paths)
    verify_archive(paths)
    final_report = validate_manuscript.validate(require_archive=True)
    print(f"[arxiv] wrote {OUTPUT}")
    print(f"  members: {len(paths)}")
    print(f"  sha256: {final_report['archive_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
