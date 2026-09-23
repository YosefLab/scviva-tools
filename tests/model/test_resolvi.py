import numpy as np
import pytest
from scvi.data import synthetic_iid

from scviva.model._resolvi import ResolVI


def _prepare_adata_with_neighbors(n_neighs=10):
    """Create a synthetic AnnData with spatial coords and neighbor arrays."""
    adata = synthetic_iid()
    n = adata.n_obs
    rng = np.random.default_rng(42)
    adata.obsm["spatial"] = rng.random((n, 2))
    adata.obs["cell_area"] = rng.gamma(2.0, 1.0, size=n)

    # Pre-compute neighbor arrays (index + distance) manually so tests
    # do not require squidpy or scanpy.
    index_neighbor = np.zeros((n, n_neighs), dtype=np.int64)
    distance_neighbor = np.ones((n, n_neighs), dtype=np.float32)
    for i in range(n):
        neighbors = rng.choice([j for j in range(n) if j != i], size=n_neighs, replace=False)
        index_neighbor[i] = neighbors
        distance_neighbor[i] = rng.uniform(0.1, 2.0, size=n_neighs)

    adata.obsm["index_neighbor"] = index_neighbor
    adata.obsm["distance_neighbor"] = distance_neighbor
    return adata


@pytest.fixture(scope="module")
def resolvi_adata():
    return _prepare_adata_with_neighbors()


def test_resolvi_setup_anndata(resolvi_adata):
    """setup_anndata registers the AnnDataManager without errors."""
    ResolVI.setup_anndata(resolvi_adata)
    assert resolvi_adata is not None


def test_resolvi_train(resolvi_adata):
    """Model initialises and trains for 2 epochs without error."""
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.train(max_epochs=1)
    assert model.is_trained


def test_resolvi_compute_neighbors(resolvi_adata):
    """compute_neighbors (squidpy backend) populates index/distance obsm keys."""
    pytest.importorskip("squidpy")
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.compute_neighbors(resolvi_adata, spatial_key="spatial", n_neighs=5)
    assert "index_neighbor" in resolvi_adata.obsm
    assert "distance_neighbor" in resolvi_adata.obsm


@pytest.mark.optional
def test_resolvi_neighbor_abundance():
    """get_neighbor_abundance returns shape (n_obs, n_cell_types) with no NaNs.

    Uses a fresh adata (not the shared fixture) to avoid mutation from
    test_resolvi_compute_neighbors, which replaces index_neighbor with n_neighs=5.
    compute_dataset_dependent_priors requires n_neighbors >= 6.
    """
    adata = _prepare_adata_with_neighbors(n_neighs=10)
    ResolVI.setup_anndata(adata, labels_key="labels")
    model = ResolVI(adata, semisupervised=True)  # probs_prediction requires semisupervised
    model.train(max_epochs=1)
    result = model.get_neighbor_abundance(return_numpy=True)
    assert result.ndim == 2
    assert result.shape[0] == adata.n_obs
    assert not np.isnan(result).any()


def test_resolvi_train_validation_unsupported():
    # RESOLVI trains with Pyro SVI and per-cell global parameters, so it does not support a
    # validation set. `train_size != 1.0` previously raised a cryptic TypeError (collision with
    # the hardcoded `train_size=1.0`), and `early_stopping` has no validation set to monitor.
    # Both must now raise a clear ValueError, while the supported settings still train.
    # Fresh adata (not the shared fixture): test_resolvi_compute_neighbors mutates it to
    # n_neighs=5, and compute_dataset_dependent_priors requires n_neighbors >= 6.
    adata = _prepare_adata_with_neighbors()
    ResolVI.setup_anndata(adata)
    model = ResolVI(adata)
    with pytest.raises(ValueError, match="train_size"):
        model.train(max_epochs=2, train_size=0.8)
    with pytest.raises(ValueError, match="early_stopping"):
        model.train(max_epochs=2, early_stopping=True)
    # explicit train_size=1.0 must not collide, and the default path still works
    model.train(max_epochs=2, train_size=1.0)


def test_resolvi_normalized_expression_gene_list():
    # Fresh adata (not the shared fixture): test_resolvi_compute_neighbors mutates it to
    # n_neighs=5, and compute_dataset_dependent_priors requires n_neighbors >= 6.
    adata = _prepare_adata_with_neighbors()
    ResolVI.setup_anndata(adata, size_factor_key="cell_area")
    model = ResolVI(adata)
    model.train(
        max_epochs=2,
    )
    gene_list = adata.var_names[:3].tolist()

    # both functions must honor `gene_list` and return only the requested subset
    expr = model.get_normalized_expression(n_samples=2, gene_list=gene_list)
    assert list(expr.columns) == gene_list
    assert expr.shape == (adata.n_obs, len(gene_list))

    expr_imp = model.get_normalized_expression_importance(n_samples=30, gene_list=gene_list)
    assert list(expr_imp.columns) == gene_list
    assert expr_imp.shape == (adata.n_obs, len(gene_list))

    # `transform_batch` is not supported by the importance estimator: it must be ignored
    # (not error) and warn the user, so that `differential_expression(weights="importance")`
    # can still call it.
    with pytest.warns(UserWarning, match="transform_batch.*ignored"):
        model.get_normalized_expression_importance(n_samples=30, transform_batch=0)


def _exact_squared_knn(coords, n_neighbors):
    from sklearn.neighbors import NearestNeighbors

    # kneighbors() without X excludes each query point itself.
    dist, idx = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors()
    return dist**2, idx


def test_squared_knn_from_distance_graph_unsorted_rows_with_self():
    """Regression test for scverse/scvi-tools#3977.

    Rows stored in column order (not distance order), with explicit self-loops and a
    coincident pair of cells, must still yield the true kNN with squared distances.
    """
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    from scviva.model.base._neighborhood_mixin import _squared_knn_from_distance_graph

    rng = np.random.default_rng(0)
    n, k = 300, 10
    coords = rng.uniform(0, 100, (n, 2))
    coords[1] = coords[0]  # distinct cells sharing a centroid (distance 0, not self)

    # kNN graph with an explicit self entry (distance 0), as scanpy's CPU path stores.
    nn_dist, nn_idx = NearestNeighbors(n_neighbors=k + 5).fit(coords).kneighbors()
    nn_idx = np.hstack([np.arange(n)[:, None], nn_idx])
    nn_dist = np.hstack([np.zeros((n, 1)), nn_dist])
    graph = sp.csr_matrix(
        (nn_dist.ravel(), nn_idx.ravel(), np.arange(0, n * (k + 6) + 1, k + 6)), shape=(n, n)
    )
    graph.sort_indices()  # canonical CSR: rows ordered by column index, not distance
    assert graph.has_sorted_indices

    dist, idx = _squared_knn_from_distance_graph(graph, k)
    true_dist, true_idx = _exact_squared_knn(coords, k)

    assert dist.shape == idx.shape == (n, k)
    assert not (idx == np.arange(n)[:, None]).any()
    assert np.all(np.diff(dist, axis=1) >= 0)
    assert idx[0, 0] == 1
    assert idx[1, 0] == 0
    assert dist[0, 0] == 0.0
    np.testing.assert_allclose(dist, true_dist, rtol=1e-5, atol=1e-6)
    overlap = np.mean([len(set(idx[i]) & set(true_idx[i])) / k for i in range(n)])
    assert overlap > 0.99  # only exact distance ties at the k-th neighbor may differ


def test_squared_knn_from_distance_graph_too_few_neighbors():
    import scipy.sparse as sp

    from scviva.model.base._neighborhood_mixin import _squared_knn_from_distance_graph

    graph = sp.csr_matrix(np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]))
    with pytest.raises(ValueError, match="fewer than the requested"):
        _squared_knn_from_distance_graph(graph, 3)


def test_resolvi_prepare_data_matches_exact_knn():
    """Regression test for scverse/scvi-tools#3977: ``_prepare_data`` stores true spatial
    kNN (per batch, no self) with squared Euclidean distances."""
    pytest.importorskip("scanpy")
    pytest.importorskip("sklearn")

    adata = synthetic_iid()
    rng = np.random.default_rng(0)
    adata.obsm["X_spatial"] = rng.uniform(0, 100, (adata.n_obs, 2)).astype(np.float32)
    k = 10
    ResolVI._prepare_data(adata, n_neighbors=k, batch_key="batch")

    idx = adata.obsm["index_neighbor"]
    dist = adata.obsm["distance_neighbor"]
    assert not (idx == np.arange(adata.n_obs)[:, None]).any()
    assert np.all(np.diff(dist, axis=1) >= 0)

    batches = adata.obs["batch"].to_numpy()
    for b in np.unique(batches):
        cells = np.where(batches == b)[0]
        assert np.all(batches[idx[cells]] == b)
        true_dist, true_idx = _exact_squared_knn(adata.obsm["X_spatial"][cells], k)
        np.testing.assert_allclose(dist[cells], true_dist, rtol=1e-4, atol=1e-3)
        overlap = np.mean(
            [len(set(idx[c]) & set(cells[true_idx[j]])) / k for j, c in enumerate(cells)]
        )
        assert overlap > 0.99


def test_resolvi_setup_anndata_recomputes_stale_neighbors():
    """Neighbors cached before the scverse/scvi-tools#3977 fix (no ``_version``) are recomputed."""
    pytest.importorskip("scanpy")

    adata = synthetic_iid()
    rng = np.random.default_rng(0)
    adata.obsm["X_spatial"] = rng.uniform(0, 100, (adata.n_obs, 2)).astype(np.float32)
    stale = np.zeros((adata.n_obs, 10), dtype=int)
    adata.obsm["index_neighbor"] = stale
    adata.obsm["distance_neighbor"] = np.ones((adata.n_obs, 10))
    adata.uns["_resolvi_prepare_data_config"] = {"batch_key": None}

    ResolVI.setup_anndata(adata)

    assert not np.array_equal(adata.obsm["index_neighbor"], stale)
    assert adata.uns["_resolvi_prepare_data_config"]["_version"] >= 2


def _spatial_adata():
    adata = synthetic_iid()
    rng = np.random.default_rng(0)
    adata.obsm["X_spatial"] = rng.uniform(0, 100, (adata.n_obs, 2)).astype(np.float32)
    return adata


def test_resolvi_setup_anndata_keeps_compute_neighbors():
    """Neighbors from ``compute_neighbors`` are kept by the default ``setup_anndata``."""
    pytest.importorskip("squidpy")
    adata = _spatial_adata()
    ResolVI.compute_neighbors(adata, spatial_key="X_spatial", n_neighs=7)
    idx = adata.obsm["index_neighbor"].copy()

    ResolVI.setup_anndata(adata)

    np.testing.assert_array_equal(adata.obsm["index_neighbor"], idx)
    assert "_resolvi_prepare_data_config" not in adata.uns
    ResolVI(adata)  # registered neighbors are usable


def test_resolvi_setup_anndata_prepare_data_kwargs_overrides_compute_neighbors():
    pytest.importorskip("squidpy")
    pytest.importorskip("scanpy")
    adata = _spatial_adata()
    ResolVI.compute_neighbors(adata, spatial_key="X_spatial", n_neighs=7)

    ResolVI.setup_anndata(adata, prepare_data_kwargs={"n_neighbors": 10})

    assert adata.obsm["index_neighbor"].shape == (adata.n_obs, 10)
    assert "_spatial_compute_neighbors_config" not in adata.uns
    assert adata.uns["_resolvi_prepare_data_config"]["n_neighbors"] == 10


def test_resolvi_setup_anndata_warns_compute_neighbors_cross_batch():
    pytest.importorskip("squidpy")
    adata = _spatial_adata()
    ResolVI.compute_neighbors(adata, spatial_key="X_spatial", n_neighs=7)
    with pytest.warns(UserWarning, match="across batches"):
        ResolVI.setup_anndata(adata, batch_key="batch")


def test_resolvi_setup_anndata_warns_before_overwriting_unmarked_neighbors():
    pytest.importorskip("scanpy")
    adata = _spatial_adata()
    adata.obsm["index_neighbor"] = np.zeros((adata.n_obs, 10), dtype=int)
    adata.obsm["distance_neighbor"] = np.ones((adata.n_obs, 10))
    with pytest.warns(UserWarning, match="Overwriting existing"):
        ResolVI.setup_anndata(adata)


def test_resolvi_neighbor_paths_agree():
    """``compute_neighbors`` and ``_prepare_data`` follow the same convention."""
    pytest.importorskip("squidpy")
    pytest.importorskip("scanpy")
    a1, a2 = _spatial_adata(), _spatial_adata()
    ResolVI.compute_neighbors(a1, spatial_key="X_spatial", n_neighs=10)
    ResolVI._prepare_data(a2, n_neighbors=10)

    np.testing.assert_allclose(
        a1.obsm["distance_neighbor"], a2.obsm["distance_neighbor"], rtol=1e-4, atol=1e-3
    )
    overlap = np.mean(
        [
            len(set(a1.obsm["index_neighbor"][i]) & set(a2.obsm["index_neighbor"][i])) / 10
            for i in range(a1.n_obs)
        ]
    )
    assert overlap > 0.99
