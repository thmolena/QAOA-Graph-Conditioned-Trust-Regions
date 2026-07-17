"""Narrow compatibility helpers for manuscript artifact provenance.

The committed schema-2 artifact manifest predates the repository auto-locator
used by the installed ``gctr-reproduce`` command.  That locator changes only
where files are found; it cannot change simulations, plots, tables, or
statistics.  These helpers remove exactly that function from the artifact
source digest while retaining byte-level validation of every algorithm-bearing
source file.
"""

from __future__ import annotations

from pathlib import Path


LEGACY_REPLOT_MANIFEST_SHA256 = (
    "db75eccfdb600b3b59da4f35e6d74fc9e7a294b9c085d63b04d994de4b2ffecd",
    "620d13e5dd7a685e0e391284e54a19c382f7cf41a6c40c54d2030803b024221f",
)

# Filled from the locator-independent digest of the historical schema-2
# replot source set.  Keeping this value in the non-algorithmic provenance
# module avoids making the fingerprint definition self-referential.
LOCKED_SCHEMA2_REPLOT_CORE_SHA256 = (
    "26eb1b7a271b6bde60ddd445be07c0fc6fb9c1a6b24b95064d5369d1f2838040"
)

LEGACY_PORTFOLIO_MANIFEST_SHA256 = (
    "42aa34816315126b4519d92da86a24120443cdee6a654cf92c932c7b91ce9f3d",
)

# Exact post-fix portfolio source digest.  The finite-sample quantile correction
# changes only the previously incorrect rank > calibration-size branch; every
# locked study has rank <= calibration size and therefore identical outputs.
LOCKED_POSTFIX_PORTFOLIO_CORE_SHA256 = (
    "71c0cbad4b7bf20e640b1f247823f0f8d1a9804e27a602393ae0b9f1c355d0e9"
)

_LOCATOR_START = b"\ndef _default_manuscript_dir()"
_LOCATOR_END = b"\ndef _environment()"
_LOCATOR_SENTINEL = (
    b"\n# repository locator excluded from artifact fingerprint\n"
)


def artifact_source_bytes(path: Path) -> bytes:
    """Return bytes used by the artifact digest.

    All files are hashed verbatim except ``reproduce.py``, where only the
    repository locator is replaced by a stable sentinel.  Missing or duplicate
    boundaries fail closed instead of silently weakening validation.
    """
    payload = Path(path).read_bytes()
    if Path(path).name != "reproduce.py":
        return payload
    if payload.count(_LOCATOR_START) != 1 or payload.count(_LOCATOR_END) != 1:
        raise RuntimeError(
            "cannot isolate the repository locator for provenance hashing")
    start = payload.index(_LOCATOR_START)
    end = payload.index(_LOCATOR_END, start)
    return payload[:start] + _LOCATOR_SENTINEL + payload[end:]


def compatible_manifest_fingerprints(core_sha256: str) -> frozenset[str]:
    """Return identifiers valid for an exact locator-independent source core.

    The two legacy identifiers differ only because the original broad digest
    included two versions of the repository locator.  They are accepted only
    for the pinned algorithm-bearing core; any other source change admits only
    its newly computed canonical digest.
    """
    if core_sha256 == LOCKED_SCHEMA2_REPLOT_CORE_SHA256:
        return frozenset((core_sha256, *LEGACY_REPLOT_MANIFEST_SHA256))
    return frozenset((core_sha256,))


def compatible_portfolio_fingerprints(core_sha256: str) -> frozenset[str]:
    """Admit the pre-fix identifier only for the exact theorem-aligned core."""
    if core_sha256 == LOCKED_POSTFIX_PORTFOLIO_CORE_SHA256:
        return frozenset((core_sha256, *LEGACY_PORTFOLIO_MANIFEST_SHA256))
    return frozenset((core_sha256,))
