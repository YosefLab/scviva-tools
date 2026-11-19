"""Small shared statistics/array helpers.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).

Kept in their own module (rather than inlined into ``preprocessing/filters.py``
or ``preprocessing/normalization.py``) to avoid circular imports:
``preprocessing/filters.py`` needs ``_get_mean_var`` but
``preprocessing/normalization.py`` needs ``log2p1``, which
``tools/projections.py`` also needs, and ``tools/projections.py`` in turn
imports ``apply_filters`` from ``preprocessing/filters.py``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)


def _looks_log_transformed(X: np.ndarray | sparse.spmatrix, max_value: float = 30.0) -> bool:
    """Heuristically decide whether *X* already looks log-transformed.

    Two signals are used; either one calling the data "not yet
    log-transformed" is enough to say so:

    1. The maximum value exceeds *max_value*. ``log2(x + 1)`` compresses even
       a very highly expressed gene (raw counts in the hundreds of
       thousands, or CPM/TPM-style normalized values up to ~1e6) down to
       roughly 15-20; genuinely log-transformed expression data essentially
       never exceeds this range.
    2. The stored (non-zero) values are, within floating-point tolerance,
       integers -- the signature of raw or lightly-processed counts, which
       are never what R VISION's ``matLog2``/this port's ``log2p1`` produce.

    This is a heuristic, not a guarantee: it can be fooled by, e.g.,
    already-log-transformed data with one pathologically large outlier, or
    by raw counts from extremely shallow sequencing that happen to stay
    below *max_value*. Pass ``check_log_transformed=False`` to :func:`log2p1`
    to bypass it when you know better.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    max_value : float, optional
        Values above this are assumed to be on a pre-log scale, by default 30.

    Returns
    -------
    bool
        ``True`` if *X* already looks log-transformed (``log2p1`` should be
        a no-op), ``False`` if it looks like it still needs the transform.
    """
    data = X.data if sparse.issparse(X) else np.asarray(X)
    if data.size == 0:
        return True  # nothing to transform either way

    max_val = float(np.max(data))
    if max_val > max_value:
        return False

    return not np.allclose(data, np.round(data), atol=1e-6)


def log2p1(
    X: np.ndarray | sparse.spmatrix,
    *,
    check_log_transformed: bool = True,
) -> np.ndarray | sparse.spmatrix:
    """Sparse-preserving log2(x + 1) transform.

    For sparse matrices only the stored (non-zero) values are transformed,
    keeping the sparsity pattern intact.  This is valid because
    ``log2(0 + 1) == 0``.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.  Not modified in-place.
    check_log_transformed : bool, optional
        If ``True`` (default), first check whether *X* already looks
        log-transformed (:func:`_looks_log_transformed`) and skip the
        transform if so. A warning is logged whenever the transform *is*
        applied, since it mutates the caller's data.

        R VISION's documented contract (and this port's, everywhere else
        that calls ``log2p1``) is that expression data should be
        scaled/normalized but **not** log-transformed -- VISION applies the
        log step itself. This check exists so that data following the
        opposite convention (already log-transformed, as is common
        after standard ``sc.pp.normalize_total`` + ``sc.pp.log1p``
        preprocessing) isn't silently log-transformed a second time.
        Pass ``False`` to always apply the transform unconditionally.

    Returns
    -------
    array-like of shape (n_cells, n_genes)
        Log-transformed matrix in the same format as the input (or an
        unchanged copy, if *X* was detected as already log-transformed).
    """
    if check_log_transformed:
        if _looks_log_transformed(X):
            logger.debug(
                "log2p1: input already looks log-transformed (small, "
                "non-integer values) -- not applying log2(x + 1) again."
            )
            return X.copy() if sparse.issparse(X) else np.asarray(X, dtype=float)
        logger.warning(
            "log2p1: input does not look log-transformed (large and/or "
            "integer-valued values found) -- applying log2(x + 1) "
            "internally. VISION expects scaled/library-size-normalized but "
            "NOT log-transformed expression data; pass "
            "check_log_transformed=False to always apply this transform "
            "without the check, or pre-transform your data and this "
            "message will stop appearing."
        )

    if sparse.issparse(X):
        X = X.copy()
        X.data = np.log2(X.data + 1)
    else:
        X = np.log2(np.asarray(X, dtype=float) + 1)
    return X


def _get_mean_var(X: np.ndarray | sparse.spmatrix, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance of *X* along *axis*, supporting sparse input.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Dense or sparse expression matrix.
    axis : int, optional
        Axis along which to compute the statistics, by default 0.

    Returns
    -------
    mean : np.ndarray
        Mean along *axis*.
    var : np.ndarray
        Variance along *axis*.
    """
    if sparse.issparse(X):
        n = X.shape[axis]
        mean = np.array(X.mean(axis=axis)).ravel()
        # Population variance (E[X^2] - mean^2), Bessel-corrected to sample
        # variance (ddof=1) to match the dense path below.
        var = np.array(X.power(2).mean(axis=axis)).ravel() - mean**2
        if n > 1:
            var *= n / (n - 1)
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis, ddof=1)
    return mean, var


def gene_centered_svd(
    X: np.ndarray, n_components: int, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gene-mean-center a dense (cells x genes) matrix and run randomized SVD.

    Shared core of :func:`~scviva.tools.vision.tools.projections.apply_pca`
    and micro-clustering's internal PCA — both center by per-gene means
    before a truncated randomized SVD.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Dense expression matrix.
    n_components : int
        Number of singular components to compute.
    random_state : int, optional
        Seed for :func:`sklearn.utils.extmath.randomized_svd`, by default 0.

    Returns
    -------
    U : ndarray of shape (n_cells, n_components)
    S : ndarray of shape (n_components,)
    Vt : ndarray of shape (n_components, n_genes)
    """
    mu = X.mean(axis=0)
    X_c = X - mu
    U, S, Vt = randomized_svd(X_c, n_components=n_components, random_state=random_state)
    return U, S, Vt
