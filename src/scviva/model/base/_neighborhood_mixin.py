from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("squidpy", "rapids")

# Version of the ``index_neighbor``/``distance_neighbor`` convention. Bump whenever the stored
# neighbors change, so arrays cached by an older release are recomputed
# (v2: squared distances, distance-sorted, self excluded -- scverse/scvi-tools#3977).
NEIGHBORS_VERSION = 2
# ``adata.uns`` key recording that neighbors were computed by :meth:`compute_neighbors`.
COMPUTE_NEIGHBORS_UNS_KEY = "_spatial_compute_neighbors_config"
# ``adata.uns`` key recording the config of the last ``ResolVI._prepare_data`` run.
RESOLVI_PREPARE_DATA_UNS_KEY = "_resolvi_prepare_data_config"


def _squared_knn_from_distance_graph(graph, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract the ``n_neighbors`` nearest neighbors per row of a sparse kNN distance graph.

    Returns element-wise *squared* Euclidean distances, sorted ascending per row, with
    self-loops removed. Rows of the input graph are not assumed to be distance-ordered.

    Parameters
    ----------
    graph
        Sparse ``(n_obs, n_obs)`` matrix of Euclidean distances, e.g.
        ``adata.obsp["distances"]`` from :func:`scanpy.pp.neighbors`.
    n_neighbors
        Number of neighbors to keep per row.

    Returns
    -------
    Tuple of squared distances and neighbor indices, both of shape ``(n_obs, n_neighbors)``.
    """
    import scipy.sparse as sp

    graph = sp.csr_matrix(graph)
    n_obs = graph.shape[0]
    rows = np.repeat(np.arange(n_obs), np.diff(graph.indptr))
    # Drop self-loops by index rather than by value: distinct cells can share a centroid.
    keep = graph.indices != rows
    rows, cols, dist = rows[keep], graph.indices[keep], np.asarray(graph.data[keep])

    counts = np.bincount(rows, minlength=n_obs)
    if counts.min() < n_neighbors:
        raise ValueError(
            f"Some cells have only {counts.min()} spatial neighbors in the distance graph, "
            f"fewer than the requested n_neighbors={n_neighbors}."
        )

    # Sort by row, then distance, then column index (for deterministic tie-breaking).
    order = np.lexsort((cols, dist, rows))
    cols, dist = cols[order], dist[order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    take = starts[:, None] + np.arange(n_neighbors)
    return dist[take] ** 2, cols[take]


class SpatialNeighborhoodMixin:
    """Mixin for spatial neighbor graph computation.

    Applied to: SCVIVA, ResolVI.
    Provides a single entry point for neighbor graph computation with
    pluggable backends (squidpy on CPU, RAPIDS on GPU).

    The computed neighbor arrays are stored in:
    - ``adata.obsm["index_neighbor"]``    — dense int array, shape (n_cells, n_neighs)
    - ``adata.obsm["distance_neighbor"]`` — dense float array, shape (n_cells, n_neighs)

    Row ``i`` holds the ``n_neighs`` nearest cells to cell ``i`` (excluding ``i`` itself),
    sorted by distance, and the corresponding **squared** Euclidean distances. This is the
    convention ResolVI's diffusion kernel expects, and the one produced by
    ``ResolVI._prepare_data``.
    """

    @classmethod
    def compute_neighbors(
        cls,
        adata: AnnData,
        spatial_key: str = "spatial",
        coord_type: str = "generic",
        n_neighs: int = 6,
        backend: Literal["squidpy", "rapids"] = "squidpy",
    ) -> None:
        """Compute spatial neighbor graph and store in ``adata.obsm``.

        Can be called on the class (e.g. ``ResolVI.compute_neighbors(adata)``) before
        ``setup_anndata``, or on a model instance. Neighbors are computed over all cells in
        ``adata`` (not per batch). ``ResolVI.setup_anndata`` keeps them instead of recomputing,
        unless ``prepare_data_kwargs`` is passed.

        Parameters
        ----------
        adata
            AnnData object with spatial coordinates in ``adata.obsm[spatial_key]``.
        spatial_key
            Key in ``adata.obsm`` for spatial coordinates.
        coord_type
            Coordinate type passed to squidpy (``"generic"`` or ``"visium"``).
            Ignored when ``backend="rapids"``.
        n_neighs
            Number of nearest neighbors.
        backend
            ``"squidpy"`` (default): uses :func:`squidpy.gr.spatial_neighbors`.
            ``"rapids"``: uses cuGraph/cuML for GPU-accelerated computation.
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got '{backend}'.")
        if backend == "squidpy":
            cls._compute_neighbors_squidpy(adata, spatial_key, coord_type, n_neighs)
        else:
            cls._compute_neighbors_rapids(adata, spatial_key, n_neighs)
        adata.uns[COMPUTE_NEIGHBORS_UNS_KEY] = {
            "spatial_key": spatial_key,
            "coord_type": coord_type,
            "n_neighs": n_neighs,
            "backend": backend,
            "_version": NEIGHBORS_VERSION,
        }
        # The arrays no longer come from ``ResolVI._prepare_data``.
        adata.uns.pop(RESOLVI_PREPARE_DATA_UNS_KEY, None)

    @staticmethod
    def _compute_neighbors_squidpy(
        adata: AnnData,
        spatial_key: str,
        coord_type: str,
        n_neighs: int,
    ) -> None:
        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError(
                "squidpy is required for backend='squidpy'. "
                "Install with: pip install 'scviva-tools[spatial]'"
            ) from e

        sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            coord_type=coord_type,
            n_neighs=n_neighs,
            key_added="spatial_neighbors",
        )

        import scipy.sparse as sp

        # Use the connectivity graph only for *which* cells are neighbors, and recompute the
        # distances from the coordinates: squidpy's distance matrix drops explicit zeros, which
        # would silently lose neighbors that share a centroid.
        conn = sp.csr_matrix(adata.obsp["spatial_neighbors_connectivities"])
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        rows = np.repeat(np.arange(adata.n_obs), np.diff(conn.indptr))
        dist = np.linalg.norm(coords[rows] - coords[conn.indices], axis=1)
        graph = sp.csr_matrix((dist, conn.indices, conn.indptr), shape=conn.shape)
        dst, idx = _squared_knn_from_distance_graph(graph, n_neighs)

        adata.obsm["index_neighbor"] = idx.astype(np.int64)
        adata.obsm["distance_neighbor"] = dst.astype(np.float32)
        logger.info("Computed %d spatial neighbors (squidpy backend).", n_neighs)

    @staticmethod
    def _compute_neighbors_rapids(
        adata: AnnData,
        spatial_key: str,
        n_neighs: int,
    ) -> None:
        try:
            import cuml
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "backend='rapids' requires cuml and cupy. "
                "Install with: pip install 'scviva-tools[rapids]'"
            ) from e

        coords = cp.asarray(adata.obsm[spatial_key].astype(np.float32))
        import scipy.sparse as sp

        nn = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighs + 1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        distances = cp.asnumpy(distances).astype(np.float64)
        indices = cp.asnumpy(indices).astype(np.int64)
        # Self is not guaranteed to be column 0 when cells share a centroid, so drop it by
        # index (via the shared helper) rather than slicing off the first column.
        n, k = indices.shape
        graph = sp.csr_matrix(
            (distances.ravel(), indices.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n)
        )
        dst, idx = _squared_knn_from_distance_graph(graph, n_neighs)
        adata.obsm["index_neighbor"] = idx.astype(np.int64)
        adata.obsm["distance_neighbor"] = dst.astype(np.float32)
        logger.info("Computed %d spatial neighbors (RAPIDS backend).", n_neighs)

    def _setup_neighbor_field(self, adata: AnnData) -> list:
        """Return neighbor obsm fields for registration in AnnDataManager.

        Call this inside setup_anndata after computing neighbors. Returns
        a list of NeighborhoodGraphField instances to include in the
        AnnDataManager fields list.
        """
        from scviva.data._fields import NeighborhoodGraphField

        for key in ("index_neighbor", "distance_neighbor"):
            if key not in adata.obsm:
                raise KeyError(
                    f"'{key}' not found in adata.obsm. "
                    "Call model.compute_neighbors(adata) before setup_anndata."
                )
        return [
            NeighborhoodGraphField(obsm_key="index_neighbor"),
            NeighborhoodGraphField(obsm_key="distance_neighbor"),
        ]
