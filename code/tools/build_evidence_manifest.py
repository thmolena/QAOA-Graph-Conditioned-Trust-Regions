#!/usr/bin/env python3
"""Write the deterministic manifest consumed by gctr-reproduce-all replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    code = Path(__file__).resolve().parents[1]
    package = code / "src/gctr_repro"
    bundle = package / "bundle"
    files = {}
    for path in sorted(
        item
        for item in bundle.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    ):
        relative = path.relative_to(bundle).as_posix()
        files[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    body = {"schema_version": 1, "files": files}
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = package / "evidence_manifest.json"
    destination.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[evidence-manifest] {len(files)} files -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
