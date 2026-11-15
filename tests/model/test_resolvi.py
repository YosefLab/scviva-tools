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


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_resolvi_prepare_data_matches_spatial_knn(batch_key):
    """prepare_data stores the true k nearest spatial neighbors with squared distances.

    Regression test: the scanpy distance graph used to be squared with ``**``, which on a
    sparse matrix is the matrix power (neighbors of neighbors), not element-wise.
    """
    from sklearn.neighbors import NearestNeighbors

    k = 10
    adata = synthetic_iid()
    coords = np.random.default_rng(0).random((adata.n_obs, 2))
    adata.obsm["X_spatial"] = coords.copy()
    ResolVI.setup_anndata(
        adata, batch_key=batch_key, prepare_data=True, prepare_data_kwargs={"n_neighbors": k}
    )
    np.testing.assert_array_equal(adata.obsm["X_spatial"], coords)

    if batch_key is None:
        groups = [np.arange(adata.n_obs)]
    else:
        groups = [np.where(adata.obs[batch_key] == b)[0] for b in adata.obs[batch_key].unique()]
    for idx in groups:
        ind = adata.obsm["index_neighbor"][idx]
        dist = adata.obsm["distance_neighbor"][idx]
        assert np.isin(ind, idx).all()  # neighbors stay within the batch
        assert (ind != idx[:, None]).all()  # no self-neighbors

        true_sq = ((coords[ind] - coords[idx][:, None, :]) ** 2).sum(-1)
        np.testing.assert_allclose(dist, true_sq, rtol=1e-4, atol=1e-7)

        # tie-robust kNN check: every stored neighbor is within the true k-th distance
        knn_dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(coords[idx]).kneighbors()
        kth_sq = knn_dist[:, k - 1] ** 2
        assert (true_sq <= kth_sq[:, None] * (1 + 1e-6)).all()
