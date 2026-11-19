"""Cell micro-pooling (supercell) construction.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).
Mirrors R VISION's ``Microclusters.R``.

Workflow
--------
1. ``apply_micro_clustering``: log2 → filter genes → PCA → KNN → Louvain
   → K-means readjustment → returns a pool mapping (pool_id → cell names).
2. ``pool_matrix``: averages expression within each pool via sparse multiply.
3. ``pool_metadata``: pools ``adata.obs`` metadata (mean for numerics,
   majority vote for categoricals).

Tree-based clustering functions (``depthBasedTreeCluster``, etc.) are part of
PhyloVision and are not ported here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from scviva.tools.vision._utils import gene_centered_svd, log2p1
from scviva.tools.vision.preprocessing.filters import apply_filters, filter_genes_threshold

from .knn import find_knn

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

# pool mapping: cluster_id → list of obs_names
Pools = dict[str, list[str]]


# ---------------------------------------------------------------------------
# Internal: Louvain clustering on exponential-kernel KNN graph
# ---------------------------------------------------------------------------


def _louvain_cluster(data: np.ndarray, K: int = 30, random_state: int = 0) -> list[list[int]]:
    """Cluster cells via Louvain on an exponential-kernel KNN graph.

    Mirrors R VISION's ``louvainCluster``.  The per-cell bandwidth σ_i equals
    the **median** K-NN distance (R: ``quantile(d_i, 0.5)``), in contrast to
    :func:`~scviva.tools.vision.tools.knn.compute_knn_weights` which uses the max
    distance.

    Parameters
    ----------
    data : ndarray of shape (n_cells, n_dims)
        Cell coordinates in latent space.
    K : int, optional
        Number of neighbors for the KNN graph, by default 30.
    random_state : int, optional
        Seed for the Louvain community detection, by default 0. igraph's
        ``community_multilevel`` has no seed parameter of its own and is
        otherwise non-deterministic between runs.

    Returns
    -------
    list of list of int
        Each inner list holds 0-based cell indices belonging to one cluster.

    Raises
    ------
    ImportError
        If ``python-igraph`` is not installed.
    """
    import random

    try:
        import igraph
    except ImportError as e:
        raise ImportError(
            "python-igraph is required for Louvain clustering. "
            "Install it with: pip install python-igraph"
        ) from e

    n_cells = data.shape[0]
    K_eff = min(K, n_cells - 1)

    idx, d = find_knn(data, K_eff)  # (n_cells, K_eff) each

    # Per-cell bandwidth: median K-NN distance (not max — differs from knn.py)
    sigma = np.median(d, axis=1, keepdims=True)  # (n_cells, 1)
    sigma[sigma == 0] = 1.0  # guard: co-located neighbors

    W = np.exp(-(d**2) / (sigma**2))  # (n_cells, K_eff)

    # Build directed edge list via vectorized index math (avoids Python loops)
    src = np.repeat(np.arange(n_cells), K_eff)  # (n_cells * K_eff,)
    dst = idx.ravel()  # (n_cells * K_eff,)
    edges = np.column_stack([src, dst]).tolist()  # igraph accepts list of [i, j]
    weights = W.ravel().tolist()

    g = igraph.Graph(n=n_cells, edges=edges, directed=True)
    g.es["weight"] = weights
    # Mirror R's as.undirected(mode="each"): sum weights when i→j and j→i coexist
    g = g.as_undirected(combine_edges="sum")

    igraph.set_random_number_generator(random.Random(random_state))
    cl = g.community_multilevel(weights=g.es["weight"])
    return [list(members) for members in cl]


# ---------------------------------------------------------------------------
# Internal: K-means readjustment to hit target partition count
# ---------------------------------------------------------------------------


def _readjust_clusters(
    clusters: list[list[int]],
    data: np.ndarray,
    cells_per_partition: int = 100,
    random_state: int = 0,
) -> list[list[int]]:
    """Refine Louvain clusters with K-means until target partition count is met.

    Mirrors R VISION's ``readjust_clusters``.  While the number of clusters is
    below ``(1 − 0.15) × target``, each cluster larger than
    *cells_per_partition* is split into ``round(size / cells_per_partition)``
    sub-clusters via K-means; smaller clusters are kept intact.  The loop
    terminates early if a pass produces no new clusters.

    Parameters
    ----------
    clusters : list of list of int
        Initial clusters; each inner list holds 0-based cell indices.
    data : ndarray of shape (n_cells, n_dims)
        Latent-space coordinates used for K-means splitting.
    cells_per_partition : int, optional
        Target cells per micro-cluster, by default 100.
    random_state : int, optional
        Seed for the K-means splitting, by default 0 (previously unseeded --
        each split was non-deterministic between runs).

    Returns
    -------
    list of list of int
        Refined clusters.
    """
    n_cells = data.shape[0]
    target = max(1, round(n_cells / cells_per_partition))
    epsilon = 0.15

    while len(clusters) < (1 - epsilon) * target:
        new_clusters: list[list[int]] = []
        for cell_idx in clusters:
            sub_data = data[cell_idx]
            if len(cell_idx) > cells_per_partition:
                n_sub = max(2, round(len(cell_idx) / cells_per_partition))
                km = KMeans(
                    n_clusters=n_sub, max_iter=100, n_init=1, random_state=random_state
                ).fit(sub_data)
                labels = km.labels_
            else:
                labels = np.zeros(len(cell_idx), dtype=int)

            for lbl in np.unique(labels):
                new_clusters.append([cell_idx[k] for k in np.where(labels == lbl)[0]])

        if len(new_clusters) == len(clusters):
            break  # no progress
        clusters = new_clusters

    return clusters


# ---------------------------------------------------------------------------
# Internal: lightweight PCA for micro-clustering
# ---------------------------------------------------------------------------


def _internal_pca(
    adata: AnnData,
    filter_input: bool,
    filter_threshold: int,
    filter_num_mad: float,
    ncomp: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """Compute a lightweight PCA for micro-clustering via randomized SVD.

    Mirrors R VISION's internal ``rsvd`` call on the gene-gene covariance
    matrix inside ``applyMicroClustering``.  The Python implementation runs
    ``randomized_svd`` on the centered data matrix directly — algebraically
    equivalent (right singular vectors of X_c equal the eigenvectors of
    X_c Xᵀ_c) but numerically simpler.

    Returns
    -------
    ndarray of shape (n_cells, ncomp)
        Cell scores (first ``ncomp`` PCs).
    """
    X_log = log2p1(adata.X)

    if filter_input:
        mask = apply_filters(
            X_log,
            adata.var_names,
            filters=("threshold", "fano"),
            threshold=filter_threshold,
            num_mad=filter_num_mad,
        )
        # Fall back to threshold-only if fano leaves too few genes
        if mask.sum() < 2:
            mask = filter_genes_threshold(X_log, threshold=filter_threshold)
        X_sub = X_log[:, mask]
    else:
        X_sub = X_log

    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray().astype(np.float32)
    else:
        X_sub = np.asarray(X_sub, dtype=np.float32)

    n_cells, n_genes = X_sub.shape
    k = min(ncomp, n_cells - 1, n_genes - 1)

    # Per-gene-mean centering (mirrors R's rowMeans(fexpr)) + randomized SVD.
    U, S, _ = gene_centered_svd(X_sub, k, random_state=random_state)
    return U * S  # (n_cells, k) — cell scores


# ---------------------------------------------------------------------------
# Public: main micro-clustering entry point
# ---------------------------------------------------------------------------


def apply_micro_clustering(
    adata: AnnData,
    cells_per_partition: int = 10,
    filter_input: bool = True,
    filter_threshold: int | None = None,
    filter_num_mad: float = 2.0,
    latent_space_key: str | None = None,
    K: int | None = None,
    uns_key: str = "micro_clusters",
    random_state: int = 0,
) -> Pools:
    """Partition cells into micro-clusters (supercells) via Louvain + K-means.

    Mirrors R VISION's ``applyMicroClustering``.

    If *latent_space_key* is provided and present in ``adata.obsm``, those
    coordinates are used directly.  Otherwise a lightweight internal PCA
    (max 10 components) is computed on log-transformed, filtered expression.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cells_per_partition : int, optional
        Target number of cells per micro-cluster, by default 10.
    filter_input : bool, optional
        Apply threshold + fano gene filters before internal PCA,
        by default ``True``.  Ignored when *latent_space_key* is provided.
    filter_threshold : int, optional
        Minimum expressing-cell count for the threshold filter. Defaults to
        ``round(0.05 * n_cells)`` (matching R VISION's ``applyMicroClustering``
        default of 5% of cells) when ``None``.
    filter_num_mad : float, optional
        MAD multiplier for the fano filter, by default 2.0.
    latent_space_key : str, optional
        Key in ``adata.obsm`` for pre-computed latent-space coordinates.
        When ``None`` (default), coordinates are computed internally.
    K : int, optional
        Number of KNN neighbors for the Louvain graph.  Defaults to
        ``min(round(sqrt(n_cells)), 30)``, matching R VISION (no lower
        floor -- R's own formula has none either).
    uns_key : str, optional
        Key under which the pool mapping is stored in ``adata.uns``,
        by default ``"micro_clusters"``.
    random_state : int, optional
        Seed for the internal PCA, Louvain clustering, and K-means
        readjustment steps, by default 0. Passing the same value across runs
        makes the whole micro-clustering pipeline reproducible.

    Returns
    -------
    dict mapping str → list[str]
        ``pools["microcluster_<i>"]`` is the list of ``obs_names`` belonging
        to that micro-cluster.
    """
    n_cells = adata.n_obs
    obs_names = list(adata.obs_names)

    if filter_threshold is None:
        filter_threshold = max(1, round(0.05 * n_cells))

    # 1. Latent space
    if latent_space_key is not None and latent_space_key in adata.obsm:
        latent = np.asarray(adata.obsm[latent_space_key], dtype=float)
        logger.info(
            "Using pre-computed latent space '%s' — shape %s.", latent_space_key, latent.shape
        )
    else:
        logger.info("Computing internal PCA for micro-clustering...")
        latent = _internal_pca(
            adata, filter_input, filter_threshold, filter_num_mad, random_state=random_state
        )

    # 2. KNN → Louvain
    if K is None:
        K = min(round(np.sqrt(n_cells)), 30)

    logger.info(
        "Louvain micro-clustering: n_cells=%d, K=%d, target ~%d cells/partition.",
        n_cells,
        K,
        cells_per_partition,
    )
    clusters = _louvain_cluster(latent, K=K, random_state=random_state)

    # 3. K-means readjustment
    clusters = _readjust_clusters(clusters, latent, cells_per_partition, random_state=random_state)
    logger.info("Micro-clustering complete: %d micro-clusters.", len(clusters))

    # 4. Map indices → obs_names and store
    pools: Pools = {
        f"microcluster_{i + 1}": [obs_names[j] for j in cell_idx]
        for i, cell_idx in enumerate(clusters)
    }
    adata.uns[uns_key] = pools
    return pools


# ---------------------------------------------------------------------------
# Pool expression matrix
# ---------------------------------------------------------------------------


def pool_matrix(
    X: np.ndarray | sparse.spmatrix,
    pools: Pools,
    obs_names: list[str] | np.ndarray,
) -> np.ndarray:
    """Average expression within each micro-cluster via sparse matrix multiply.

    Mirrors R VISION's ``poolMatrixCols_Inner``.  Builds a binary
    ``n_cells × n_pools`` indicator matrix and multiplies it against the
    expression matrix in a single sparse operation, then divides each row by
    its pool size.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.
    pools : dict
        Mapping ``pool_id → list[obs_name]`` as returned by
        :func:`apply_micro_clustering`.
    obs_names : sequence of str
        Cell names aligned to rows of *X*.

    Returns
    -------
    ndarray of shape (n_pools, n_genes)
        Per-pool mean expression.
    """
    obs_index = {name: i for i, name in enumerate(obs_names)}
    n_cells = X.shape[0]
    pool_ids = list(pools.keys())
    n_pools = len(pool_ids)

    # Build indicator matrix indices via vectorized array ops (avoids inner loop)
    sizes = np.array([len(pools[pid]) for pid in pool_ids], dtype=np.intp)
    cell_idx_lists = [
        np.fromiter((obs_index[n] for n in pools[pid]), dtype=np.intp, count=sz)
        for pid, sz in zip(pool_ids, sizes, strict=True)
    ]
    rows = np.concatenate(cell_idx_lists)
    cols = np.repeat(np.arange(n_pools, dtype=np.intp), sizes)

    # n_cells × n_pools binary indicator
    pool_mat = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_cells, n_pools),
    )

    # (n_pools × n_cells) @ (n_cells × n_genes) → n_pools × n_genes
    pooled = pool_mat.T @ X
    if sparse.issparse(pooled):
        pooled = pooled.toarray()
    else:
        pooled = np.asarray(pooled, dtype=float)

    pooled /= np.array(sizes, dtype=float)[:, None]
    return pooled


def pool_matrix_anndata(
    adata: AnnData,
    pools: Pools | None = None,
    layer_key: str | None = None,
    uns_key: str = "micro_clusters",
    result_uns_key: str = "pool_expression",
) -> np.ndarray:
    """Pool ``adata.X`` (or a layer) into per-micro-cluster mean expression.

    Convenience wrapper around :func:`pool_matrix` that reads pools from
    ``adata.uns`` when *pools* is not provided explicitly.

    Parameters
    ----------
    adata : AnnData
    pools : dict, optional
        Pool mapping.  Reads from ``adata.uns[uns_key]`` when ``None``.
    layer_key : str, optional
        Layer to pool.  Uses ``adata.X`` when ``None``.
    uns_key : str
        Key in ``adata.uns`` for the pool mapping.
    result_uns_key : str
        Key in ``adata.uns`` where the result is stored.

    Returns
    -------
    ndarray of shape (n_pools, n_genes)
    """
    if pools is None:
        pools = adata.uns[uns_key]
    X = adata.layers[layer_key] if layer_key else adata.X
    pooled = pool_matrix(X, pools, list(adata.obs_names))
    adata.uns[result_uns_key] = pooled
    return pooled


# ---------------------------------------------------------------------------
# Pool metadata
# ---------------------------------------------------------------------------


def pool_metadata(
    obs: pd.DataFrame,
    pools: Pools,
) -> pd.DataFrame:
    """Build a pooled metadata table for micro-clusters.

    Mirrors R VISION's ``poolMetaData``:

    - **Numeric columns**: per-pool mean.
    - **Categorical / string columns**: majority level when ≥ 50 % of cells
      agree; otherwise ``"~"``.  Per-level proportion columns
      ``<Var>_<Level>`` are appended for every level observed *anywhere in
      ``obs``* (not just within that pool) -- a pool that happens not to
      contain a given level gets an explicit ``0.0`` for it, matching R,
      which always writes every ``levels(factor)`` entry for every pool.

    Parameters
    ----------
    obs : pd.DataFrame of shape (n_cells, n_vars)
        Cell-level metadata (``adata.obs``).
    pools : dict
        Mapping ``pool_id → list[obs_name]``.

    Returns
    -------
    pd.DataFrame of shape (n_pools, ≥n_vars)
        Pooled metadata indexed by pool IDs.
    """
    cat_cols = [c for c in obs.columns if not pd.api.types.is_numeric_dtype(obs[c])]
    all_levels = {c: obs[c].unique() for c in cat_cols}

    rows = []
    for _pid, cell_names in pools.items():
        sub = obs.loc[cell_names]
        row: dict = {}

        for col in sub.columns:
            if pd.api.types.is_numeric_dtype(sub[col]):
                row[col] = float(sub[col].mean())
            else:
                counts = sub[col].value_counts()
                top_level = counts.index[0]
                row[col] = top_level if counts.iloc[0] / len(sub) >= 0.5 else "~"
                for level in all_levels[col]:
                    row[f"{col}_{level}"] = float((sub[col] == level).mean())

        rows.append(row)

    return pd.DataFrame(rows, index=list(pools.keys()))


def pool_metadata_anndata(
    adata: AnnData,
    pools: Pools | None = None,
    uns_key: str = "micro_clusters",
    result_uns_key: str = "pool_metadata",
) -> pd.DataFrame:
    """Pool ``adata.obs`` into per-micro-cluster metadata.

    Convenience wrapper around :func:`pool_metadata`.

    Parameters
    ----------
    adata : AnnData
    pools : dict, optional
        Pool mapping.  Reads from ``adata.uns[uns_key]`` when ``None``.
    uns_key : str
        Key in ``adata.uns`` for the pool mapping.
    result_uns_key : str
        Key in ``adata.uns`` where the result is stored.

    Returns
    -------
    pd.DataFrame of shape (n_pools, ≥n_obs_vars)
    """
    if pools is None:
        pools = adata.uns[uns_key]
    result = pool_metadata(adata.obs, pools)
    adata.uns[result_uns_key] = result
    return result
