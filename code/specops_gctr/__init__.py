"""specops-gctr: Query-Efficient QAOA via Graph-Conditioned Trust Regions.

Part of the "spectral-truncation operators" program (Molena Huynh). This
package is a REAL end-to-end pipeline: exact-statevector QAOA on MaxCut, a
torch GNN predicting a graph-conditioned Gaussian over angles, a held-out
error calibrator that allocates per-instance search budgets, and
query-efficiency / calibration / cross-size / leave-one-family-out / ablation
/ seed-stability / shot-noise studies. The `gctr-reproduce` entrypoint
regenerates every data figure, table and source-data CSV in the manuscript.
BibTeX key: huynh2026gctr.
"""
__version__ = "3.0.0"

from .pipeline import Config  # noqa: F401
