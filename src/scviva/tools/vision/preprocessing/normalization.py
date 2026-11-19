"""Expression matrix normalization methods.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).
Mirrors R VISION's ``NormalizationMethods.R``.

Orientation note
----------------
R VISION stores expression as genes × cells, so R's "columns" are cells and
R's "rows" are genes.  Python / AnnData uses cells × genes, which flips the
axes.  The method name strings (``"znorm_columns"``, ``"znorm_rows"``, …) are
kept identical to R so that existing VISION configuration values work without
change; the docstrings clarify what each string normalises in the Python layout.

Available method strings
------------------------
``"none"``                  No normalization (identity).
``"znorm_columns"``         Z-normalize each cell across genes
                            (R: per-column = per-cell; Python: per-row).
``"znorm_rows"``            Z-normalize each gene across cells
                            (R: per-row = per-gene; Python: per-column).
``"znorm_rows_then_columns"`` Z-normalize genes first, then cells.
``"rank_norm_columns"``     Replace each cell's expression values with their
                            within-cell min-rank
                            (R: per-column ranks; Python: per-row ranks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from scviva.tools.vision._utils import _get_mean_var, log2p1

logger = logging.getLogger(__name__)

NormMethod = Literal[
    "none",
    "znorm_columns",
    "znorm_rows",
    "znorm_rows_then_columns",
    "rank_norm_columns",
]

_VALID_METHODS = {
    "none",
    "znorm_columns",
    "znorm_rows",
    "znorm_rows_then_columns",
    "rank_norm_columns",
}
_SPARSE_METHODS = {"none", "znorm_columns", "znorm_rows", "znorm_rows_then_columns"}


# ---------------------------------------------------------------------------
# Dense helper normalisers
# ---------------------------------------------------------------------------


def _normalize_per_cell(X: np.ndarray) -> np.ndarray:
    """Z-normalize each cell (row in Python / column in R) across genes.

    Parameters
    ----------
    X : np.ndarray of shape (n_cells, n_genes)
        Dense expression matrix.

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes)
        Per-cell z-normalized matrix.
    """
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, ddof=1, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma


def _normalize_per_gene(X: np.ndarray) -> np.ndarray:
    """Z-normalize each gene (column in Python / row in R) across cells.

    Parameters
    ----------
    X : np.ndarray of shape (n_cells, n_genes)
        Dense expression matrix.

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes)
        Per-gene z-normalized matrix.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma


def _normalize_rank_per_cell(X: np.ndarray) -> np.ndarray:
    """Replace each cell's values with within-cell min-ranks.

    Mirrors R's ``colRankNormalization`` (``colRanks(..., ties.method="min")``).

    Parameters
    ----------
    X : np.ndarray of shape (n_cells, n_genes)
        Dense expression matrix.

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes)
        Matrix of within-cell min-ranks.
    """
    return rankdata(X, method="min", axis=1).astype(float)


# ---------------------------------------------------------------------------
# Dense end-to-end normalisation (getNormalizedCopy equivalent)
# ---------------------------------------------------------------------------


def get_normalized_copy(
    X: np.ndarray | sparse.spmatrix,
    method: NormMethod = "znorm_columns",
) -> np.ndarray:
    """Log2-transform then normalise *X* using the chosen method.

    Mirrors R VISION's ``getNormalizedCopy``: always applies ``matLog2``
    (i.e. ``log2(x+1)``) first, then the chosen normalisation.  Sparse
    inputs are densified after the log transform.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Raw or pre-normalised expression matrix, cells × genes.
    method : str
        One of ``"none"``, ``"znorm_columns"``, ``"znorm_rows"``,
        ``"znorm_rows_then_columns"``, ``"rank_norm_columns"``.

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes)
        Dense normalised matrix.

    Raises
    ------
    ValueError
        If *method* is not one of the recognised strings.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Normalisation method '{method}' not recognised. "
            f"Choose from: {sorted(_VALID_METHODS)}."
        )

    X = log2p1(X)
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=float)

    if method == "none":
        return X
    elif method == "znorm_columns":
        return _normalize_per_cell(X)
    elif method == "znorm_rows":
        return _normalize_per_gene(X)
    elif method == "znorm_rows_then_columns":
        return _normalize_per_cell(_normalize_per_gene(X))
    else:  # rank_norm_columns
        return _normalize_rank_per_cell(X)


# ---------------------------------------------------------------------------
# NormData — lazy sparse-preserving normalisation (getNormalizedCopySparse)
# ---------------------------------------------------------------------------


@dataclass
class NormData:
    """Sparse expression matrix with pre-computed normalisation parameters.

    Stores the log-transformed sparse matrix together with per-gene and
    per-cell offset / scale-factor vectors so that the fully normalised
    value for element ``(i, j)`` can be computed on demand without
    materialising a dense intermediate matrix.

    Normalised value for element ``(i, j)``::

        step1 = (data[i, j] + gene_offsets[j]) * gene_scale_factors[j]
        step2 = (step1 + cell_offsets[i]) * cell_scale_factors[i]

    Parameters that are not used for a given method are set to neutral
    values (offsets=0, scale_factors=1).

    Mirrors R VISION's ``NormData`` S4 class from ``NormalizationMethods.R``.
    Note that R uses genes × cells layout (so R's ``rowOffsets`` are
    per-gene and R's ``colOffsets`` are per-cell); the Python attribute names
    use the Python / AnnData convention (cells × genes).

    Parameters
    ----------
    data : array-like of shape (n_cells, n_genes)
        Log-transformed expression matrix, dense or sparse.
    gene_offsets : np.ndarray of shape (n_genes,)
        Per-gene offsets (R's ``rowOffsets``).
    gene_scale_factors : np.ndarray of shape (n_genes,)
        Per-gene scale factors (R's ``rowScaleFactors``).
    cell_offsets : np.ndarray of shape (n_cells,)
        Per-cell offsets (R's ``colOffsets``).
    cell_scale_factors : np.ndarray of shape (n_cells,)
        Per-cell scale factors (R's ``colScaleFactors``).
    """

    data: np.ndarray | sparse.spmatrix
    gene_offsets: np.ndarray  # shape (n_genes,) — R's rowOffsets
    gene_scale_factors: np.ndarray  # shape (n_genes,) — R's rowScaleFactors
    cell_offsets: np.ndarray  # shape (n_cells,) — R's colOffsets
    cell_scale_factors: np.ndarray  # shape (n_cells,) — R's colScaleFactors

    def toarray(self) -> np.ndarray:
        """Materialise the fully normalised dense matrix.

        Returns
        -------
        np.ndarray of shape (n_cells, n_genes)
            Fully normalised dense expression matrix.
        """
        X = (
            self.data.toarray()
            if sparse.issparse(self.data)
            else np.asarray(self.data, dtype=float)
        )
        # Apply gene normalisation (per-column in Python)
        X = (X + self.gene_offsets) * self.gene_scale_factors
        # Apply cell normalisation (per-row in Python)
        X = (X + self.cell_offsets[:, None]) * self.cell_scale_factors[:, None]
        return X


def _cell_norm_params_after_gene_norm(
    X: np.ndarray | sparse.spmatrix,
    gene_offsets: np.ndarray,
    gene_scale_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cell offsets and scale factors after gene normalization.

    Analytically derives what the per-cell means and SDs would be once
    gene normalization has been applied, without forming the dense
    intermediate matrix.  Mirrors R VISION's ``.colNormHelper``.

    After gene normalization element ``(i, j)`` equals::

        (X_ij + gene_offsets[j]) * gene_scale_factors[j]

    The per-cell mean is then::

        cell_mean_i = (X[i] @ gsf + Σ_j gof_j·gsf_j) / n_genes

    and the variance is computed by expanding the squared terms analytically
    (see source comments and R's ``.colNormHelper`` for derivation).

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Log-transformed sparse expression matrix.
    gene_offsets : ndarray of shape (n_genes,)
        Per-gene offsets (= ``-gene_means``).
    gene_scale_factors : ndarray of shape (n_genes,)
        Per-gene scale factors (= ``1 / gene_std``).

    Returns
    -------
    cell_offsets : ndarray of shape (n_cells,)
        Per-cell offsets for the second normalization step (= ``-cell_means``).
    cell_scale_factors : ndarray of shape (n_cells,)
        Per-cell scale factors (= ``1 / cell_std``); set to 1 where std == 0.
    """
    n_cells, n_genes = X.shape
    gof = gene_offsets  # (n_genes,)
    gsf = gene_scale_factors  # (n_genes,)

    # ── per-cell mean after gene normalization ──────────────────────────────
    # cell_mean_i = (Σ_j X_ij·gsf_j + Σ_j gof_j·gsf_j) / n_genes
    if sparse.issparse(X):
        xgsf_sum = np.asarray(X.multiply(gsf).sum(axis=1)).ravel()
    else:
        xgsf_sum = (X * gsf).sum(axis=1)  # (n_cells,)

    const = float(np.sum(gof * gsf))
    cell_means = (xgsf_sum + const) / n_genes  # (n_cells,)
    cell_offsets = -cell_means

    # ── per-cell variance after gene normalization ──────────────────────────
    # Var_i = [Σ_j ((X_ij + gof_j)·gsf_j)² − n_genes·cell_mean_i²] / (n_genes−1)
    # Expand the square into three computable terms:
    #   t1 = Σ_j X_ij²·gsf_j²
    #   t2 = 2·Σ_j X_ij·gof_j·gsf_j²
    #   t3 = Σ_j gof_j²·gsf_j²  (scalar, same for every cell)
    gsf2 = gsf**2  # (n_genes,)
    gof_gsf2 = gof * gsf2  # (n_genes,)

    if sparse.issparse(X):
        t1 = np.asarray(X.power(2).multiply(gsf2).sum(axis=1)).ravel()
        t2 = 2.0 * np.asarray(X.multiply(gof_gsf2).sum(axis=1)).ravel()
    else:
        t1 = ((X**2) * gsf2).sum(axis=1)
        t2 = 2.0 * (X * gof_gsf2).sum(axis=1)

    t3 = float(np.sum((gof**2) * gsf2))
    sum_sq = t1 + t2 + t3  # (n_cells,)
    cell_var = (sum_sq - n_genes * cell_means**2) / max(n_genes - 1, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cell_scale_factors = np.where(cell_var > 0, cell_var ** (-0.5), 1.0)

    return cell_offsets, cell_scale_factors


def get_normalized_copy_sparse(
    X: np.ndarray | sparse.spmatrix,
    method: NormMethod = "znorm_columns",
) -> NormData:
    """Log2-transform then build a sparse-preserving :class:`NormData` object.

    Mirrors R VISION's ``getNormalizedCopySparse``.  Instead of materialising
    the normalised matrix, the offsets and scale factors are pre-computed and
    stored in a :class:`NormData` instance.  Call :meth:`NormData.toarray` to
    obtain the dense result when needed.

    ``"rank_norm_columns"`` is not supported in the sparse path (it must
    densify; use :func:`get_normalized_copy` instead).

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    method : str
        One of ``"none"``, ``"znorm_columns"``, ``"znorm_rows"``,
        ``"znorm_rows_then_columns"``.

    Returns
    -------
    NormData
        Lazy normalisation container.

    Raises
    ------
    ValueError
        If *method* is not supported by the sparse path.
    """
    if method not in _SPARSE_METHODS:
        raise ValueError(
            f"Method '{method}' is not supported in the sparse path. "
            f"Choose from: {sorted(_SPARSE_METHODS)}, or use get_normalized_copy()."
        )

    X = log2p1(X)
    n_cells, n_genes = X.shape

    gene_offsets = np.zeros(n_genes, dtype=float)
    gene_scale_factors = np.ones(n_genes, dtype=float)
    cell_offsets = np.zeros(n_cells, dtype=float)
    cell_scale_factors = np.ones(n_cells, dtype=float)

    if method == "znorm_rows" or method == "znorm_rows_then_columns":
        # Z-normalize genes (per-column in Python / per-row in R)
        gene_means, gene_var = _get_mean_var(X, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gsfact = np.where(gene_var > 0, gene_var ** (-0.5), 1.0)
        gene_offsets = -gene_means
        gene_scale_factors = gsfact

    if method == "znorm_columns":
        # Z-normalize cells (per-row in Python / per-column in R)
        cell_means, cell_var = _get_mean_var(X, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            csfact = np.where(cell_var > 0, cell_var ** (-0.5), 1.0)
        cell_offsets = -cell_means
        cell_scale_factors = csfact

    if method == "znorm_rows_then_columns":
        # Analytically derive cell stats *after* gene normalization without
        # forming the dense intermediate — mirrors R's .colNormHelper.
        cell_offsets, cell_scale_factors = _cell_norm_params_after_gene_norm(
            X, gene_offsets, gene_scale_factors
        )

    return NormData(
        data=X,
        gene_offsets=gene_offsets,
        gene_scale_factors=gene_scale_factors,
        cell_offsets=cell_offsets,
        cell_scale_factors=cell_scale_factors,
    )
