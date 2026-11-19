"""Latent space and 2D projection computation for VISION.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).

Mirrors R VISION's ``Projections.R`` and the PCA section of
``AnalysisFunctions.R``:
  - sparse-preserving log2(x+1) transform
  - PCA via randomized truncated SVD with gene-mean centering
  - permutation significance test for automatic PC selection (Buja & Eyuboglu 1992)
  - end-to-end ``compute_latent_space`` that filters → transforms → runs PCA
    and stores results in ``adata.obsm``
  - 2D visualisation projections: tSNE (perplexity 10 and 30), UMAP (via
    scanpy), ICA (via sklearn FastICA), ISOMap (via sklearn)
  - ``generate_projections`` orchestrator that mirrors ``generateProjectionsInner``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import norm

from scviva.tools.vision._utils import gene_centered_svd, log2p1
from scviva.tools.vision.preprocessing.filters import apply_filters

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)

_NUM_PERMUTATION_REPEATS = 20


def _single_permutation(seed: int, X: np.ndarray, n_components: int) -> np.ndarray:
    """One column-permutation replicate for apply_permutation_wpca.

    Must be a module-level function so joblib loky can pickle it.

    Parameters
    ----------
    seed : int
        Seed for the per-replicate random number generator.
    X : ndarray of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    n_components : int
        Number of PCs to compute for the permuted matrix.

    Returns
    -------
    ndarray of shape (n_components,)
        Variance-explained values from the permuted PCA.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = X.shape
    perm_idx = np.argsort(rng.random((n_cells, n_genes)), axis=0)
    X_perm = X[perm_idx, np.arange(n_genes)]
    _, eval_perm, _ = apply_pca(X_perm, n_components)
    return eval_perm


# ---------------------------------------------------------------------------
# Core PCA
# ---------------------------------------------------------------------------


def apply_pca(
    X: np.ndarray | sparse.spmatrix,
    max_components: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via randomized truncated SVD with gene-mean centering.

    Mirrors R VISION's ``applyPCA`` which runs IRLBA on
    ``t(exprData)`` with ``center = rowMeans(exprData)``.  The Python
    equivalent works on the (cells × genes) layout and centers by
    subtracting per-gene means before the SVD.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.  Sparse matrices are densified
        internally because centering destroys sparsity.
    max_components : int, optional
        Maximum number of PCs to compute, by default 200.

    Returns
    -------
    pca_coords : ndarray of shape (n_cells, n_components)
        Cell coordinates in PC space (scores).
    variance_explained : ndarray of shape (n_components,)
        Proportion of total variance captured by each PC.
    gene_loadings : ndarray of shape (n_genes, n_components)
        Right singular vectors (variable loadings) for each PC.

    Notes
    -----
    ``total_var`` is computed as ``sum((X - mu)²)`` which equals
    R's ``sum(exprData²) - ncol(exprData) * sum(rM²)`` — algebraically
    identical expressions for the variance of the centered matrix.
    """
    if sparse.issparse(X):
        X = X.toarray().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    n_cells, n_genes = X.shape
    n_components = min(max_components, n_cells - 1, n_genes - 1)

    # Center by gene means (mirrors R's center=rowMeans(exprData) in irlba) and
    # run randomized SVD (equivalent to IRLBA for large matrices).
    U, S, Vt = gene_centered_svd(X, n_components)

    pca_coords = U * S  # (n_cells, n_components)
    mu = X.mean(axis=0)
    total_var = np.sum(X**2) - n_cells * np.sum(mu**2)  # sum((X - mu)**2)
    variance_explained = S**2 / total_var  # proportion per PC
    gene_loadings = Vt.T  # (n_genes, n_components)

    return pca_coords, variance_explained, gene_loadings


# ---------------------------------------------------------------------------
# Permutation WPCA
# ---------------------------------------------------------------------------


def apply_permutation_wpca(
    X: np.ndarray | sparse.spmatrix,
    max_components: int = 50,
    p_threshold: float = 0.05,
    n_repeats: int = _NUM_PERMUTATION_REPEATS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA with permutation significance test for automatic PC selection.

    Based on the method of Buja & Eyuboglu (1992).  Each gene's expression
    is independently permuted across cells ``n_repeats`` times; the resulting
    variance-explained values form a null distribution.  A z-score is computed
    for each real PC against the null, and PCs are retained while their
    p-value (from the normal survival function) is below *p_threshold*.
    A minimum of 5 PCs is always kept.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Log-transformed expression matrix, cells × genes.
    max_components : int, optional
        Maximum number of PCs to evaluate, by default 50.
    p_threshold : float, optional
        P-value cut-off for PC retention, by default 0.05.
    n_repeats : int, optional
        Number of permutation replicates, by default 20.

    Returns
    -------
    pca_coords : ndarray of shape (n_cells, n_sig)
        Cell scores for the retained PCs.
    variance_explained : ndarray of shape (n_sig,)
        Proportion of variance explained by each retained PC.
    gene_loadings : ndarray of shape (n_genes, n_sig)
        Gene loadings for each retained PC.

    Notes
    -----
    The R implementation slices ``evec[1:k, ]`` (rows) when it should slice
    ``evec[, 1:k]`` (columns) to subset PCs from the loading matrix.  This is
    a bug in the R source; the Python implementation applies the correct column
    slice on ``gene_loadings``.
    """
    if sparse.issparse(X):
        X = X.toarray().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    n_cells, n_genes = X.shape
    n_components = min(max_components, n_cells - 1, n_genes - 1)

    # Real PCA
    pca_coords, eval_real, gene_loadings = apply_pca(X, n_components)
    n_components = len(eval_real)  # may be less than requested

    # Build null distribution by permuting each gene independently, in parallel.
    # Each replicate independently shuffles each gene (column) across cells.
    logger.info(
        "Running %d permutation replicates for PC significance testing (parallel)...",
        n_repeats,
    )

    seeds = np.random.default_rng(0).integers(0, 2**31, size=n_repeats)
    bg_vals = np.array(
        Parallel(n_jobs=-1, backend="loky")(
            delayed(_single_permutation)(int(s), X, n_components) for s in seeds
        )
    )

    mu_bg = bg_vals.mean(axis=0)
    sigma_bg = bg_vals.std(axis=0, ddof=1)
    sigma_bg[sigma_bg == 0] = 1.0  # guard against zero-variance null

    # P-value from normal survival function (1 - CDF)
    z_scores = (eval_real - mu_bg) / sigma_bg
    pvals = 1.0 - norm.cdf(z_scores)

    # Retain PCs up to (but not including) the first non-significant one
    above_threshold = np.where(pvals > p_threshold)[0]
    n_sig = int(above_threshold[0]) if len(above_threshold) > 0 else n_components
    n_sig = max(n_sig, 5)  # always keep at least 5 components

    logger.info(
        "Permutation WPCA: %d / %d PCs significant (p < %.2f).",
        n_sig,
        n_components,
        p_threshold,
    )

    return pca_coords[:, :n_sig], eval_real[:n_sig], gene_loadings[:, :n_sig]


# ---------------------------------------------------------------------------
# End-to-end latent space computation
# ---------------------------------------------------------------------------


def compute_latent_space(
    adata: AnnData,
    norm_data_key: str | None = None,
    filters: Sequence[Literal["novar", "threshold", "fano"]] = ("threshold", "fano"),
    threshold: int = 10,
    num_mad: float = 2.0,
    max_components: int = 30,
    use_permutation_wpca: bool = False,
    projection_genes: Sequence[str] | None = None,
    obsm_key: str = "X_pca",
) -> AnnData:
    """Filter genes, log-transform, and compute PCA; store result in ``adata.obsm``.

    Mirrors R VISION's ``computeLatentSpace``, which orchestrates gene
    filtering, ``matLog2`` transform, and either ``applyPCA`` or
    ``applyPermutationWPCA``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    norm_data_key : str, optional
        Key in ``adata.layers`` to use as expression input.  If ``None``,
        ``adata.X`` is used.
    filters : sequence of str, optional
        Gene filters to apply before PCA, by default ``("threshold", "fano")``.
        Ignored when *projection_genes* is provided.
    threshold : int, optional
        Minimum number of cells for threshold / fano filters, by default 10.
    num_mad : float, optional
        MAD multiplier for the fano filter, by default 2.0.
    max_components : int, optional
        Maximum number of PCs to compute, by default 30.
    use_permutation_wpca : bool, optional
        If ``True``, run permutation WPCA to select only statistically
        significant PCs.  If ``False`` (default), return all *max_components*
        PCs (mirrors R's ``perm_wPCA=FALSE`` default).
    projection_genes : sequence of str, optional
        Explicit list of gene names to use.  When provided, *filters* are
        skipped.  Genes not found in ``adata.var_names`` are silently dropped.
    obsm_key : str, optional
        Key under which to store the PC coordinates in ``adata.obsm``,
        by default ``"X_pca"``.

    Returns
    -------
    AnnData
        The input *adata* modified in-place with:

        - ``adata.obsm[obsm_key]`` : ndarray (n_cells × n_sig_pcs)
        - ``adata.uns["pca"]["variance_ratio"]`` : proportion variance per PC
        - ``adata.uns["pca"]["gene_loadings"]`` : ndarray (n_genes × n_sig_pcs)
        - ``adata.uns["pca"]["projection_genes"]`` : list of gene names used

    Raises
    ------
    ValueError
        If no genes remain after filtering / intersection.
    """
    logger.info("Computing latent space for expression data...")

    # ------------------------------------------------------------------
    # 1. Retrieve expression
    # ------------------------------------------------------------------
    if norm_data_key is None:
        X = adata.X
    else:
        X = adata.layers[norm_data_key]

    gene_names = np.asarray(adata.var_names)

    # ------------------------------------------------------------------
    # 2. Select projection genes
    # ------------------------------------------------------------------
    if projection_genes is not None:
        provided = np.asarray(projection_genes)
        mask = np.isin(gene_names, provided)
        n_match = mask.sum()
        logger.info(
            "Using supplied gene list: %d / %d genes found in adata.",
            n_match,
            len(provided),
        )
        if n_match == 0:
            raise ValueError(
                "None of the supplied projection_genes were found in adata.var_names."
            )
    else:
        logger.info("Determining projection genes via filters %s...", list(filters))
        # Apply log2(x+1) before filtering (mirrors R's matLog2 in computeProjectionGenes)
        X_log = log2p1(X)
        mask = apply_filters(
            X_log, gene_names, filters=filters, threshold=threshold, num_mad=num_mad
        )
        if mask.sum() == 0:
            raise ValueError(
                f"Filtering with filters={list(filters)}, threshold={threshold} "
                "produced 0 genes. Lower the threshold and re-run."
            )

    selected_genes = gene_names[mask].tolist()
    logger.info("Projection genes selected: %d.", len(selected_genes))

    # ------------------------------------------------------------------
    # 3. Log-transform the filtered expression matrix
    # ------------------------------------------------------------------
    X_sub = X[:, mask]
    X_log = log2p1(X_sub)

    # ------------------------------------------------------------------
    # 4. Run PCA
    # ------------------------------------------------------------------
    if use_permutation_wpca:
        pca_coords, variance_ratio, gene_loadings = apply_permutation_wpca(
            X_log, max_components=max_components
        )
    else:
        pca_coords, variance_ratio, gene_loadings = apply_pca(X_log, max_components=max_components)

    # ------------------------------------------------------------------
    # 5. Store results
    # ------------------------------------------------------------------
    adata.obsm[obsm_key] = pca_coords
    adata.uns["pca"] = {
        "variance_ratio": variance_ratio,
        "gene_loadings": gene_loadings,
        "projection_genes": selected_genes,
        "params": {
            "obsm_key": obsm_key,
            "max_components": max_components,
            "use_permutation_wpca": use_permutation_wpca,
        },
    }

    logger.info(
        "Latent space stored in adata.obsm['%s'] — shape %s.",
        obsm_key,
        pca_coords.shape,
    )
    return adata


# ---------------------------------------------------------------------------
# 2D visualisation projections
# ---------------------------------------------------------------------------


def apply_rbfpca(
    X: np.ndarray,
    n_components: int = 2,
    sigma: float = 0.33,
) -> np.ndarray:
    """RBF-kernel PCA 2D projection.

    Mirrors R VISION's ``applyRBFPCA``.  Builds a log-distance Gaussian kernel
    matrix K from pairwise Euclidean distances, row-normalises and z-scores each
    row, then returns the projection onto the top ``n_components`` eigenvectors
    of ``K @ K.T``.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_dims)
        Input coordinates (latent space).
    n_components : int, optional
        Number of components to retain, by default 2.
    sigma : float, optional
        RBF bandwidth (applied to log-distances), by default 0.33 (matches R).

    Returns
    -------
    ndarray of shape (n_cells, n_components)

    Notes
    -----
    The row z-scoring step is a **deliberate deviation from what R VISION
    actually executes**, not from what it appears to intend. R's line reads
    ``kMat <- as.matrix(kMat, 1, function(x) (x - mean(x)) / sd(x))``, which
    looks like an ``apply()``-style per-row z-score -- but ``kMat`` is
    already a matrix, so ``as.matrix.default()`` returns it unchanged and
    silently discards the extra arguments. R's RBFPCA therefore never
    actually z-scores its kernel matrix. This implementation performs the
    z-score for real, matching the evident intent of that R line rather than
    R's actual (apparently unintentional) runtime behavior; the two
    projections are not numerically equivalent as a result.
    """
    from sklearn.metrics import pairwise_distances
    from sklearn.utils.extmath import randomized_svd

    D = pairwise_distances(X)  # (n_cells, n_cells) Euclidean
    # avoid log(0) on the diagonal (will be zeroed later)
    np.fill_diagonal(D, 1.0)
    np.log(D, out=D)  # log-distance
    D **= 2  # squared log-distance (point_mult in R)
    K = np.exp(-D / sigma**2)  # RBF kernel
    np.fill_diagonal(K, 0.0)  # zero diagonal (matches R)

    row_sums = K.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    K /= row_sums[:, np.newaxis]  # row-normalise

    # Z-score each row (matches R's as.matrix(kMat, 1, function(x) (x-mean(x))/sd(x)))
    K -= K.mean(axis=1, keepdims=True)
    stds = K.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    K /= stds

    W = K @ K.T  # covariance (tcrossprod in R)
    U, _, _ = randomized_svd(W, n_components=n_components, random_state=0)
    return K @ U  # project (crossprod(t(kMat), evec) in R)


_ALL_PROJECTION_METHODS = ("tSNE30", "tSNE10", "UMAP", "ISOMap", "ICA", "RBFPCA")


def apply_tsne(
    X: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 800,
) -> np.ndarray:
    """t-SNE 2D projection via sklearn.

    Mirrors R VISION's ``applytSNE10`` / ``applytSNE30``.  Input is expected
    to be the latent-space coordinates (PCA scores), matching R's behaviour of
    passing ``t(latentSpace)`` to ``Rtsne`` with ``pca=FALSE``.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_dims)
        Latent-space cell coordinates.
    perplexity : float, optional
        t-SNE perplexity, by default 30.0.  Use 10.0 for the ``"tSNE10"``
        variant.
    n_iter : int, optional
        Maximum number of optimisation iterations, by default 800
        (matches R's ``max_iter=800``).

    Returns
    -------
    ndarray of shape (n_cells, 2)
        2D t-SNE coordinates.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        init="random",  # pca=FALSE in R — skip internal PCA pre-step
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def apply_umap(
    adata: AnnData,
    obsm_key: str = "X_pca",
    K: int = 15,
) -> np.ndarray:
    """UMAP 2D projection via scanpy.

    Mirrors R VISION's ``applyUMAP`` (which uses ``uwot::umap``).  Runs
    ``scanpy.pp.neighbors`` on the latent space stored in
    ``adata.obsm[obsm_key]``, then ``scanpy.tl.umap``.  Scanpy writes its
    neighbour graph into ``adata.obsp["connectivities"]`` / ``"distances"``,
    which is separate from VISION's exponential-kernel weights stored in
    ``adata.obsp["weights"]`` — no conflict.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with latent space in ``adata.obsm[obsm_key]``.
    obsm_key : str, optional
        Key in ``adata.obsm`` for the latent-space coordinates, by default
        ``"X_pca"``.
    K : int, optional
        Number of nearest neighbours for the UMAP graph, by default 15
        (matches R's default ``K = round(sqrt(ncol(expr)))`` context).

    Returns
    -------
    ndarray of shape (n_cells, 2)
        2D UMAP coordinates (copy of ``adata.obsm["X_umap"]``).
    """
    import scanpy as sc

    sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=K)
    sc.tl.umap(adata)
    return adata.obsm["X_umap"].copy()


def apply_ica(
    X: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """ICA 2D projection via sklearn FastICA.

    Mirrors R VISION's ``applyICA`` which runs ``fastICA`` on raw
    log-transformed expression (not the PCA latent space).  Input should
    therefore be the filtered, log-transformed expression matrix
    (cells × genes), not PCA scores.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Log-transformed expression, cells × genes.
    n_components : int, optional
        Number of independent components, by default 2.

    Returns
    -------
    ndarray of shape (n_cells, n_components)
        ICA source-signal matrix (``$S`` in R fastICA output).
    """
    from sklearn.decomposition import FastICA

    ica = FastICA(
        n_components=n_components,
        max_iter=100,
        tol=1e-5,
        algorithm="parallel",
        fun="logcosh",
    )
    return ica.fit_transform(X)


def apply_isomap(
    X: np.ndarray,
    K: int | None = None,
) -> np.ndarray:
    """ISOMap 2D projection via sklearn.

    Mirrors R VISION's ``applyISOMap`` (``vegan::isomap`` with
    ``epsilon=1e6`` — effectively a fully-connected distance graph).  sklearn's
    ``Isomap`` uses a KNN-based neighbourhood graph; setting K large
    (≈ 3 × √n_cells) approximates the all-connected behaviour.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_dims)
        Latent-space cell coordinates.
    K : int, optional
        Number of neighbours.  Defaults to
        ``min(n_cells − 1, max(10, 3 · round(√n_cells)))``.

    Returns
    -------
    ndarray of shape (n_cells, 2)
        2D ISOMap coordinates.
    """
    from sklearn.manifold import Isomap

    n_cells = X.shape[0]
    if K is None:
        K = min(n_cells - 1, max(10, 3 * round(np.sqrt(n_cells))))

    isomap = Isomap(n_components=2, n_neighbors=K)
    return isomap.fit_transform(X)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_projections(
    adata: AnnData,
    projection_methods: Sequence[str] = _ALL_PROJECTION_METHODS,
    obsm_key: str = "X_pca",
    K: int | None = None,
    projection_genes: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Generate 2D visualisation projections and store in ``adata.obsm``.

    Mirrors R VISION's ``generateProjectionsInner``.  Dispatches each
    method to the appropriate input:

    - ``"ICA"`` — filtered log-transformed expression (cells × genes), not
      the latent space, to match R VISION's behaviour.
    - ``"UMAP"`` — scanpy neighbour graph built on ``adata.obsm[obsm_key]``.
    - ``"tSNE30"``, ``"tSNE10"``, ``"ISOMap"``, ``"RBFPCA"`` — ``adata.obsm[obsm_key]``.

    Results are stored in ``adata.obsm[f"X_{{method}}"]``
    (e.g. ``"X_tSNE30"``, ``"X_UMAP"``, ``"X_RBFPCA"``).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.  Must have ``obsm_key`` in ``adata.obsm``
        (produced by :func:`compute_latent_space`).
    projection_methods : sequence of str, optional
        Ordered list of projection names.  Choose from
        ``"tSNE30"``, ``"tSNE10"``, ``"UMAP"``, ``"ISOMap"``, ``"ICA"``,
        ``"RBFPCA"``.  Default: all six methods.
    obsm_key : str, optional
        Key in ``adata.obsm`` for the latent-space coordinates used by
        UMAP, tSNE, and ISOMap, by default ``"X_pca"``.
    K : int, optional
        Number of neighbours for UMAP and ISOMap.  Defaults to
        ``round(sqrt(n_cells))``.
    projection_genes : sequence of str, optional
        Subset of genes to use for the ICA projection.  When ``None``,
        all genes in ``adata.var_names`` are used.

    Returns
    -------
    dict mapping str → ndarray of shape (n_cells, 2)
        Same mapping as the ``adata.obsm`` entries written by this function.

    Raises
    ------
    KeyError
        If *obsm_key* is not present in ``adata.obsm``.
    ValueError
        If an unrecognised method name is requested.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in adata.obsm. Run compute_latent_space() first.")

    latent = np.asarray(adata.obsm[obsm_key], dtype=float)
    n_cells = latent.shape[0]

    if K is None:
        K = max(1, round(np.sqrt(n_cells)))

    unrecognised = set(projection_methods) - set(_ALL_PROJECTION_METHODS)
    if unrecognised:
        raise ValueError(
            f"Unrecognised projection method(s): {sorted(unrecognised)}. "
            f"Choose from: {list(_ALL_PROJECTION_METHODS)}."
        )

    results: dict[str, np.ndarray] = {}
    n_methods = len(projection_methods)

    for i, method in enumerate(projection_methods, 1):
        logger.info("Running projection %d/%d: %s...", i, n_methods, method)

        if method == "tSNE30":
            coords = apply_tsne(latent, perplexity=30.0)
        elif method == "tSNE10":
            coords = apply_tsne(latent, perplexity=10.0)
        elif method == "UMAP":
            coords = apply_umap(adata, obsm_key=obsm_key, K=K)
        elif method == "ISOMap":
            coords = apply_isomap(latent, K=K)
        elif method == "RBFPCA":
            coords = apply_rbfpca(latent)
        elif method == "ICA":
            # ICA runs on raw log-transformed expression, not the latent space
            X_raw = adata.X
            if projection_genes is not None:
                gene_mask = np.isin(adata.var_names, projection_genes)
                X_raw = X_raw[:, gene_mask]
            X_log = log2p1(X_raw)
            if sparse.issparse(X_log):
                n_dense_gb = X_log.shape[0] * X_log.shape[1] * 4 / 1e9
                if n_dense_gb > 1.0:
                    logger.warning(
                        "ICA will densify a %.1f GB sparse matrix (%d cells × %d genes). "
                        "Pass a smaller projection_genes list to reduce memory.",
                        n_dense_gb,
                        X_log.shape[0],
                        X_log.shape[1],
                    )
                X_log = X_log.toarray()
            coords = apply_ica(np.asarray(X_log, dtype=np.float32))
        else:
            continue  # should not happen given validation above

        obs_key = f"X_{method}"
        adata.obsm[obs_key] = coords
        results[method] = coords
        logger.info("  Stored in adata.obsm['%s'] — shape %s.", obs_key, coords.shape)

    return results
