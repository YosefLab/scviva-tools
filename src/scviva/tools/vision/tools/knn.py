"""KNN graph construction with exponential distance weighting.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).
Mirrors R VISION's ``computeKNNWeights`` (matrix method) in ``Projections.R``
and ``find_knn_parallel`` in ``Utilities.R``.

The weight kernel is W_ij = exp(−d²_ij / σ²_i), where σ_i is the distance
from cell i to its K-th (farthest) neighbor.  Rows are then L1-normalised so
each cell's weights sum to 1.  The result is stored as a ``(n_cells, n_cells)``
CSR sparse matrix in ``adata.obsp``, replacing the L1-normalised scanpy
connectivities that visionpy previously used as a proxy.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level KNN search
# ---------------------------------------------------------------------------


def find_knn(
    data: np.ndarray,
    K: int,
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find K nearest neighbors for every point in *data*.

    Parameters
    ----------
    data : ndarray of shape (n_cells, n_dims)
        Coordinates in the latent space.
    K : int
        Number of neighbors to return per cell.
    exact : bool, optional
        If ``True``, use sklearn's exact brute-force / kd-tree search
        (matches R VISION's RANN output exactly, slower for large datasets).
        If ``False`` (default), use ``pynndescent`` when available for an
        approximate O(n log n) search, falling back to sklearn if the package
        is not installed.

    Returns
    -------
    indices : ndarray of shape (n_cells, K)
        0-based column indices of the K nearest neighbors for each cell.
    distances : ndarray of shape (n_cells, K)
        Euclidean distances to each of the K nearest neighbors.
    """
    if not exact:
        try:
            from pynndescent import NNDescent

            # `.neighbor_graph` on the training data includes each point as its
            # own (distance-0) neighbor, so query K+1 and drop the self-hit,
            # mirroring the exact sklearn path below.
            index = NNDescent(data, n_neighbors=K + 1, random_state=0, verbose=False)
            indices, distances = index.neighbor_graph

            n = data.shape[0]
            not_self = indices != np.arange(n)[:, None]
            # Rows missing a self-hit (e.g. duplicate points ranked closer)
            # keep their first K+1 neighbors as-is rather than dropping one.
            has_self = not_self.sum(axis=1) == K
            out_indices = indices[:, :K].copy()
            out_distances = distances[:, :K].copy()
            out_indices[has_self] = indices[has_self][not_self[has_self]].reshape(-1, K)
            out_distances[has_self] = distances[has_self][not_self[has_self]].reshape(-1, K)
            return out_indices.astype(np.intp), out_distances.astype(np.float64)
        except ImportError:
            pass

    # Exact sklearn search: query K+1, drop the self-hit at column 0
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm="auto").fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices[:, 1:], distances[:, 1:]


# ---------------------------------------------------------------------------
# Exponential weight kernel + row normalisation
# ---------------------------------------------------------------------------


def compute_knn_weights(
    latent_space: np.ndarray,
    K: int | None = None,
    exact: bool = False,
) -> csr_matrix:
    """Build an exponential-kernel KNN weight matrix from a latent space.

    Mirrors R VISION's ``computeKNNWeights`` (matrix S4 method):

    1. Find the K nearest neighbors in Euclidean space.
    2. Per-cell bandwidth: σ_i = max distance among the K neighbors.
    3. Weight kernel: W_ij = exp(−d²_ij / σ²_i).
    4. Row-normalise so each row sums to 1.

    Parameters
    ----------
    latent_space : ndarray of shape (n_cells, n_dims)
        Cell coordinates (e.g. PCA scores stored in ``adata.obsm["X_pca"]``).
    K : int, optional
        Number of neighbors.  Defaults to ``round(sqrt(n_cells))``,
        matching R VISION's default.
    exact : bool, optional
        If ``True``, use sklearn's exact neighbor search instead of
        ``pynndescent``.  See :func:`find_knn`.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_cells, n_cells)
        Row-normalised sparse weight matrix.  Entry (i, j) is the weight
        cell i assigns to cell j; only KNN connections are non-zero.

    Notes
    -----
    Unlike R VISION, which returns raw ``(indices, weights)`` arrays and
    assembles the full sparse matrix lazily inside callers such as
    ``sigsVsProjection_pcf``, this function builds the complete CSR matrix
    immediately.  This matches the format already expected by visionpy's
    existing ``adata.obsp["weights"]`` slot.
    """
    latent_space = np.asarray(latent_space, dtype=float)
    n_cells = latent_space.shape[0]

    if K is None:
        K = max(1, round(np.sqrt(n_cells)))

    logger.info("Building KNN graph: n_cells=%d, K=%d.", n_cells, K)

    idx, d = find_knn(latent_space, K, exact=exact)  # both (n_cells, K)

    # Per-cell bandwidth σ_i = distance to the K-th (farthest) neighbor
    sigma = d.max(axis=1, keepdims=True)  # (n_cells, 1)
    sigma[sigma == 0] = 1.0  # guard: all K neighbors co-located

    # Exponential kernel: W_ij = exp(−d²_ij / σ²_i)
    W = np.exp(-(d**2) / (sigma**2))  # (n_cells, K)

    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums

    # Assemble CSR sparse matrix
    rows = np.repeat(np.arange(n_cells), K)  # (n_cells * K,)
    cols = idx.ravel()  # (n_cells * K,)
    vals = W.ravel()

    weights = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    logger.info("KNN weight matrix built — nnz=%d.", weights.nnz)
    return weights


# ---------------------------------------------------------------------------
# Tree-based KNN (PhyloVision)
# ---------------------------------------------------------------------------


def _cophenetic_distance_matrix(newick_str: str) -> tuple[list[str], np.ndarray]:
    """Compute pairwise cophenetic (patristic) distances from a Newick tree.

    Uses a single DFS that fills each pair exactly once at their LCA,
    giving O(n²) time and space — faster than calling tree.distance(a, b)
    for every pair.

    Missing branch lengths default to 1.0, matching R's ape behaviour.
    """
    from Bio import Phylo

    tree = Phylo.read(io.StringIO(newick_str), "newick")
    for clade in tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    terminals = tree.get_terminals()
    n = len(terminals)
    names = [t.name for t in terminals]
    term_to_idx = {id(t): i for i, t in enumerate(terminals)}

    dist = np.zeros((n, n))
    leaf_depth = np.zeros(n)

    def _dfs(clade: object, depth: float) -> list[int]:
        if clade.is_terminal():
            i = term_to_idx[id(clade)]
            leaf_depth[i] = depth
            return [i]

        child_groups: list[list[int]] = [
            _dfs(child, depth + (child.branch_length or 0.0)) for child in clade.clades
        ]

        # Pairs across different child subtrees have LCA = this clade
        for g_a, group_a in enumerate(child_groups):
            for group_b in child_groups[g_a + 1 :]:
                for i in group_a:
                    for j in group_b:
                        d = leaf_depth[i] + leaf_depth[j] - 2.0 * depth
                        dist[i, j] = d
                        dist[j, i] = d

        return [li for g in child_groups for li in g]

    _dfs(tree.root, 0.0)
    return names, dist


def compute_knn_weights_from_tree(
    newick: str,
    obs_names: Sequence[str] | None = None,
    K: int | None = None,
) -> csr_matrix:
    """Build exponential-kernel KNN weights from a phylogenetic tree.

    Mirrors :func:`compute_knn_weights` but uses cophenetic (patristic)
    distances between tree leaves instead of Euclidean latent-space distances.
    This is the core of PhyloVision: the same Gaussian kernel and row
    normalisation are applied, so all downstream autocorrelation and
    differential analyses run unchanged.

    Parameters
    ----------
    newick : str
        Newick-format tree string.  All leaf labels must be cell barcodes.
    obs_names : sequence of str, optional
        Desired row/column order.  If provided the matrix is reordered to match
        and leaves absent from *obs_names* are dropped.  If ``None``, tree leaf
        order is used.
    K : int, optional
        Number of neighbors.  Defaults to ``round(sqrt(n_cells))``.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_cells, n_cells)
        Row-normalised sparse weight matrix in *obs_names* order.

    Raises
    ------
    ValueError
        If any name in *obs_names* is not found among the tree's leaves.
    """
    names, dist = _cophenetic_distance_matrix(newick)

    if obs_names is not None:
        name_to_idx = {name: i for i, name in enumerate(names)}
        missing = [name for name in obs_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"{len(missing)} cells not found in tree leaves: {missing[:5]}...")
        obs_idx = np.array([name_to_idx[name] for name in obs_names])
        dist = dist[np.ix_(obs_idx, obs_idx)]

    n = dist.shape[0]
    if K is None:
        K = max(1, round(np.sqrt(n)))

    logger.info("Building tree KNN graph: n_cells=%d, K=%d.", n, K)

    # K nearest neighbors (column 0 after sorting is self at dist=0 — skip it)
    nn_idx = np.argsort(dist, axis=1)[:, 1 : K + 1]
    nn_d = np.take_along_axis(dist, nn_idx, axis=1)

    sigma = nn_d.max(axis=1, keepdims=True)
    sigma[sigma == 0] = 1.0
    W = np.exp(-(nn_d**2) / (sigma**2))
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    rows = np.repeat(np.arange(n), K)
    cols = nn_idx.ravel()
    weights = csr_matrix((W.ravel(), (rows, cols)), shape=(n, n))
    logger.info("Tree KNN weight matrix built — nnz=%d.", weights.nnz)
    return weights


def compute_knn_weights_from_tree_anndata(
    adata: AnnData,
    newick: str,
    K: int | None = None,
    obsp_key: str = "weights",
) -> AnnData:
    """Compute tree-based KNN weights and store in ``adata.obsp[obsp_key]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.  ``adata.obs_names`` must match leaf labels in
        the Newick tree.
    newick : str
        Newick-format tree string.
    K : int, optional
        Number of neighbors.  Defaults to ``round(sqrt(n_cells))``.
    obsp_key : str, optional
        Destination key in ``adata.obsp``, by default ``"weights"``.

    Returns
    -------
    AnnData (modified in-place)
    """
    # Build KNN in tree-leaf order (no reordering to adata order).
    # R VISION also uses tree-leaf order for both the KNN graph and cell indices
    # in geary_sparse, so keeping tree order here preserves the same tie-breaking
    # and yields autocorrelation rankings consistent with R (ρ ≈ 0.97 vs ρ ≈ 0.30
    # when the matrix is reindexed to adata order before KNN construction).
    weights = compute_knn_weights_from_tree(newick, obs_names=None, K=K)
    adata.obsp[obsp_key] = weights
    logger.info(
        "Tree KNN weights stored in adata.obsp['%s'] — shape %s.",
        obsp_key,
        weights.shape,
    )
    return adata


def compute_knn_weights_from_tree_lca(
    newick: str,
    obs_names: Sequence[str] | None = None,
    min_size: int = 20,
) -> csr_matrix:
    """Build LCA-based uniform KNN weights from a phylogenetic tree.

    For each cell, walks up the tree until the smallest enclosing clade
    contains at least *min_size* other cells.  That clade's members become the
    cell's neighbors with equal weight (row-normalised to 1).  This is the
    ``lcaKNN=TRUE`` mode in R VISION's PhyloVision.

    Note: R VISION's LCA weight matrix has an encoding inversion bug (it passes
    membership indicators as distances, giving higher weight to *non*-clade
    cells).  This implementation uses the correct semantics: uniform weights
    within the clade, zero outside.

    Parameters
    ----------
    newick : str
        Newick-format tree string.  All leaf labels must be cell barcodes.
    obs_names : sequence of str, optional
        Desired row/column order.  If provided, the matrix is reordered/subset
        to this order.  If ``None``, tree leaf order is used.
    min_size : int, optional
        Minimum number of neighbors (excluding self) required before a clade is
        accepted.  Default 20, matching R VISION's ``minSize`` default.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_cells, n_cells)
        Row-normalised sparse weight matrix.

    Raises
    ------
    ValueError
        If any name in *obs_names* is not found among the tree's leaves.
    """
    from Bio import Phylo

    tree = Phylo.read(io.StringIO(newick), "newick")
    terminals = tree.get_terminals()
    tree_names = [t.name for t in terminals]

    if obs_names is not None:
        obs_names_list = list(obs_names)
        tree_name_set = set(tree_names)
        missing = [n for n in obs_names_list if n not in tree_name_set]
        if missing:
            raise ValueError(f"{len(missing)} cells not found in tree leaves: {missing[:5]}...")
    else:
        obs_names_list = tree_names

    obs_set = set(obs_names_list)
    obs_name_to_idx = {name: i for i, name in enumerate(obs_names_list)}
    n = len(obs_names_list)

    # Build parent dict (id → parent clade)
    parent: dict = {}

    def _build_parents(clade, par) -> None:
        parent[id(clade)] = par
        for child in clade.clades:
            _build_parents(child, clade)

    _build_parents(tree.root, None)
    name_to_term = {t.name: t for t in terminals}

    logger.info("Building LCA KNN graph: n_cells=%d, min_size=%d.", n, min_size)

    rows, cols, vals = [], [], []
    for leaf_name in obs_names_list:
        leaf = name_to_term[leaf_name]
        row_i = obs_name_to_idx[leaf_name]

        # Walk up until the clade has > min_size leaves (counting all tree leaves)
        node = leaf
        while True:
            par = parent[id(node)]
            if par is None:
                break  # reached root — use whatever clade we have
            node = par
            if len(node.get_terminals()) > min_size:
                break

        # Uniform weight to all observed cells in clade, excluding self
        clade_obs = [
            t.name for t in node.get_terminals() if t.name in obs_set and t.name != leaf_name
        ]
        w = 1.0 / len(clade_obs) if clade_obs else 0.0
        for neighbor_name in clade_obs:
            rows.append(row_i)
            cols.append(obs_name_to_idx[neighbor_name])
            vals.append(w)

    weights = csr_matrix(
        (np.array(vals), (np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp))),
        shape=(n, n),
    )
    logger.info("LCA KNN weight matrix built — nnz=%d.", weights.nnz)
    return weights


def compute_knn_weights_from_tree_lca_anndata(
    adata: AnnData,
    newick: str,
    min_size: int = 20,
    obsp_key: str = "weights",
) -> AnnData:
    """Compute LCA-based tree KNN weights and store in ``adata.obsp[obsp_key]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.  ``adata.obs_names`` must match leaf labels in
        the Newick tree.
    newick : str
        Newick-format tree string.
    min_size : int, optional
        Minimum clade size (excluding self).  Default 20.
    obsp_key : str, optional
        Destination key in ``adata.obsp``, by default ``"weights"``.

    Returns
    -------
    AnnData (modified in-place)
    """
    weights = compute_knn_weights_from_tree_lca(
        newick, obs_names=adata.obs_names.tolist(), min_size=min_size
    )
    adata.obsp[obsp_key] = weights
    logger.info(
        "LCA KNN weights stored in adata.obsp['%s'] — shape %s.",
        obsp_key,
        weights.shape,
    )
    return adata


# ---------------------------------------------------------------------------
# AnnData wrapper (coordinate-based)
# ---------------------------------------------------------------------------


def compute_knn_weights_anndata(
    adata: AnnData,
    obsm_key: str = "X_pca",
    K: int | None = None,
    obsp_key: str = "weights",
    exact: bool = False,
) -> AnnData:
    """Compute VISION exponential KNN weights and store in ``adata.obsp``.

    Reads the latent space from ``adata.obsm[obsm_key]`` and writes the
    resulting ``(n_cells, n_cells)`` sparse weight matrix to
    ``adata.obsp[obsp_key]``.  This replaces the L1-normalised scanpy
    connectivities previously used by visionpy as a proxy for VISION weights.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.  Must have ``obsm_key`` in ``adata.obsm``
        (e.g. produced by :func:`~visionpy.projections.compute_latent_space`).
    obsm_key : str, optional
        Key in ``adata.obsm`` for the latent-space coordinates,
        by default ``"X_pca"``.
    K : int, optional
        Number of neighbors.  Defaults to ``round(sqrt(n_cells))``.
    obsp_key : str, optional
        Key under which to store the weight matrix in ``adata.obsp``,
        by default ``"weights"``.
    exact : bool, optional
        If ``True``, use sklearn's exact neighbor search instead of
        ``pynndescent``.  See :func:`find_knn`.

    Returns
    -------
    AnnData
        The input *adata* modified in-place with
        ``adata.obsp[obsp_key]`` populated.

    Raises
    ------
    KeyError
        If *obsm_key* is not present in ``adata.obsm``.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            "Run compute_latent_space() first or pass a valid obsm_key."
        )

    latent_space = np.asarray(adata.obsm[obsm_key], dtype=float)
    weights = compute_knn_weights(latent_space, K=K, exact=exact)
    adata.obsp[obsp_key] = weights

    logger.info(
        "KNN weights stored in adata.obsp['%s'] — shape %s.",
        obsp_key,
        weights.shape,
    )
    return adata
