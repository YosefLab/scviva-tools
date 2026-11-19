"""Gene filtering utilities for selecting projection genes.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).
Mirrors R VISION's ``Filters.R``: genes are filtered before PCA so that only
informative, overdispersed features drive the latent space.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import sparse

from scviva.tools.vision._utils import _get_mean_var

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def filter_genes_novar(
    X: np.ndarray | sparse.spmatrix,
) -> np.ndarray:
    """Remove genes with zero variance across cells.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_genes,); ``True`` for genes that pass.
    """
    _, gene_var = _get_mean_var(X, axis=0)
    passing = gene_var != 0
    logger.info("No-variance filter: %d / %d genes retained.", passing.sum(), len(passing))
    return passing


def filter_genes_threshold(
    X: np.ndarray | sparse.spmatrix,
    threshold: int = 10,
) -> np.ndarray:
    """Keep genes expressed (count > 0) in at least *threshold* cells.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    threshold : int, optional
        Minimum number of cells a gene must be expressed in, by default 10.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_genes,); ``True`` for genes that pass.
    """
    if sparse.issparse(X):
        nonzero_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        nonzero_per_gene = np.sum(X > 0, axis=0)

    passing = nonzero_per_gene >= threshold
    logger.info(
        "Threshold filter (min %d cells): %d / %d genes retained.",
        threshold,
        passing.sum(),
        len(passing),
    )
    return passing


def filter_genes_fano(
    X: np.ndarray | sparse.spmatrix,
    num_mad: float = 2.0,
    n_bins: int = 30,
    max_cells: int = 50_000,
    random_state: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Select overdispersed genes via within-bin Fano factor filtering.

    Genes are ranked by mean expression and divided into *n_bins* equal-size
    bins.  Within each bin the median and MAD of the Fano factor
    (variance / mean) are computed; genes whose Fano factor exceeds
    ``median + num_mad * MAD`` are retained.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    num_mad : float, optional
        Number of median absolute deviations above the within-bin median Fano
        required to pass, by default 2.0.
    n_bins : int, optional
        Number of mean-expression bins, by default 30.
    max_cells : int, optional
        If the dataset exceeds this many cells, subsample *max_cells* rows for
        computing gene statistics, by default 50,000.
    random_state : int, np.random.Generator, or None, optional
        Seed (or generator) for the subsampling above ``max_cells``, for
        reproducible gene selection. Defaults to ``0``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_genes,); ``True`` for genes that pass.

    Notes
    -----
    Genes with zero mean expression have their Fano factor set to 0 so they
    fall below any reasonable within-bin threshold; they are effectively
    excluded without producing NaN values.

    This is a **deliberate deviation from R VISION's actual behavior**, not
    an oversight: R's ``filterGenesFano`` computes ``fano <- var/mu`` with no
    zero-mean guard, so a single unexpressed gene produces ``0/0 = NaN``,
    which (via R's ``median()`` having no ``na.rm``) silently zeroes out
    *every gene in that gene's expression-level bin*, not just the
    zero-mean one -- R's own attempted fix for this (``mu_quant[mu_quant ==
    0] <- 1``) is itself dead code, since ``mu_quant`` is never read again
    afterwards. This implementation intentionally does not reproduce that
    bug: only the zero-mean gene itself is excluded, so Python can retain
    genes that R would silently (and unintentionally) drop.
    """
    n_cells, n_genes = X.shape

    if n_cells > max_cells:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_cells, max_cells, replace=False)
        X = X[idx]
        logger.info("Fano filter: subsampled to %d cells for statistics.", max_cells)

    mu, gene_var = _get_mean_var(X, axis=0)

    # Fano = variance / mean; guard mu == 0 (unexpressed genes) by setting fano
    # to 0 — they land in the first bin and won't exceed the threshold.
    with np.errstate(invalid="ignore", divide="ignore"):
        fano = np.where(mu > 0, gene_var / mu, 0.0)

    # Sort genes by mean expression ascending (mirrors R's `order(mu)`)
    order = np.argsort(mu)
    fano_sorted = fano[order]

    m = n_genes // n_bins  # bin width; matches R's floor(length / N_QUANTS)
    passing_sorted = np.zeros(n_genes, dtype=bool)

    for i in range(n_bins):
        if i < n_bins - 1:
            rr = slice(i * m, (i + 1) * m)
        else:
            # Last bin absorbs all remaining genes, matching R's
            # `seq(i*m+1, length(mu_sort))` special case.
            rr = slice(i * m, n_genes)

        fano_bin = fano_sorted[rr]
        if len(fano_bin) == 0:
            continue

        med = np.median(fano_bin)
        mad = np.median(np.abs(fano_bin - med))
        passing_sorted[rr] = fano_bin > (med + num_mad * mad)

    # Restore original gene order.
    # passing_sorted[i] belongs to the gene at original index order[i],
    # so `passing[order] = passing_sorted` is the inverse of argsort.
    passing = np.zeros(n_genes, dtype=bool)
    passing[order] = passing_sorted

    logger.info(
        "Fano filter (%.1f MADs, %d bins): %d / %d genes retained.",
        num_mad,
        n_bins,
        passing.sum(),
        n_genes,
    )
    return passing


def apply_filters(
    X: np.ndarray | sparse.spmatrix,
    gene_names: np.ndarray,
    filters: Sequence[Literal["novar", "threshold", "fano"]] = ("threshold", "fano"),
    threshold: int = 10,
    num_mad: float = 2.0,
) -> np.ndarray:
    """Apply one or more gene filters sequentially and return a boolean gene mask.

    Each filter in *filters* is applied in order; its result overwrites the
    previous one, mirroring R VISION's ``applyFilters`` behaviour.

    Available filters
    -----------------
    ``"novar"``
        Remove genes with zero variance across cells.
    ``"threshold"``
        Remove genes expressed in fewer than *threshold* cells.
    ``"fano"``
        Apply threshold filtering first, then retain only overdispersed genes
        using the Fano factor within mean-expression bins.  Because the fano
        filter embeds a threshold step, passing both ``"threshold"`` and
        ``"fano"`` gives the same result as passing just ``"fano"``.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    gene_names : array-like of shape (n_genes,)
        Gene identifiers corresponding to columns of *X*.  Used for logging;
        the returned mask is aligned to this ordering.
    filters : sequence of str, optional
        Ordered list of filter names to apply, by default ``("threshold", "fano")``.
    threshold : int, optional
        Minimum number of expressing cells for threshold / fano filters,
        by default 10.
    num_mad : float, optional
        MAD multiplier for the fano filter, by default 2.0.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_genes,); ``True`` for genes retained after
        all filters.

    Raises
    ------
    ValueError
        If an unrecognised filter name is provided.

    Examples
    --------
    >>> mask = apply_filters(adata.X, adata.var_names)
    >>> adata_filtered = adata[:, mask].copy()
    """
    gene_names = np.asarray(gene_names)
    n_genes = X.shape[1]
    passing = np.ones(n_genes, dtype=bool)

    for f in filters:
        f_lower = f.lower()
        if f_lower == "novar":
            passing = filter_genes_novar(X)
        elif f_lower == "threshold":
            passing = filter_genes_threshold(X, threshold=threshold)
        elif f_lower == "fano":
            # R's applyFilters applies threshold first, subsets the matrix,
            # then runs fano on the subset — both steps must pass.
            thresh_mask = filter_genes_threshold(X, threshold=threshold)
            fano_mask = filter_genes_fano(X[:, thresh_mask], num_mad=num_mad)
            passing = np.zeros(n_genes, dtype=bool)
            thresh_indices = np.where(thresh_mask)[0]
            passing[thresh_indices[fano_mask]] = True
        else:
            raise ValueError(
                f"Unrecognised filter '{f}'. Choose from 'novar', 'threshold', 'fano'."
            )

    logger.info(
        "apply_filters(%s): %d / %d genes retained.",
        list(filters),
        passing.sum(),
        n_genes,
    )
    return passing
