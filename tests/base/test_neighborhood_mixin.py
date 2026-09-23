# tests/base/test_neighborhood_mixin.py
import numpy as np
import pytest
from anndata import AnnData
from scvi.model.base import UnsupervisedTrainingMixin

from scviva.model.base._neighborhood_mixin import SpatialNeighborhoodMixin
from scviva.model.base._spatial_base import SpatialBaseModel


def _make_coords_adata(n=100, n_genes=20):
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


def test_compute_neighbors_squidpy_adds_obsm():
    pytest.importorskip("squidpy")
    adata = _make_coords_adata()

    class _M(SpatialNeighborhoodMixin, SpatialBaseModel, UnsupervisedTrainingMixin):
        def train(self, *args, **kwargs):
            pass

        @classmethod
        def setup_anndata(cls, adata, **kwargs):
            from scvi.data import AnnDataManager
            from scvi.data.fields import LayerField

            mgr = AnnDataManager(fields=[LayerField("X", "counts", is_count_data=True)])
            mgr.register_fields(adata)
            cls.register_manager(mgr)

    _M.setup_anndata(adata)
    model = _M(adata)
    model.compute_neighbors(adata, coord_type="generic", n_neighs=6, backend="squidpy")
    assert "index_neighbor" in adata.obsm
    assert "distance_neighbor" in adata.obsm
    assert adata.obsm["index_neighbor"].shape == (100, 6)
    # Guard against silent all-zero failure (wrong squidpy obsp key suffix)
    assert adata.obsm["index_neighbor"].sum() > 0, (
        "index_neighbor is all zeros — squidpy obsp key suffix may have changed. "
        "Check sq.gr.spatial_neighbors key_added convention for installed squidpy version."
    )


def test_compute_neighbors_invalid_backend():
    adata = _make_coords_adata()
    mixin = SpatialNeighborhoodMixin()
    with pytest.raises(ValueError, match="backend"):
        mixin.compute_neighbors(adata, backend="unknown")


def _coords_with_coincident_pair(n=200, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    coords[1] = coords[0]  # distinct cells sharing a centroid
    return coords


def _exact_squared_knn(coords, k):
    from sklearn.neighbors import NearestNeighbors

    # kneighbors() without X excludes each query point itself.
    dist, idx = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors()
    return dist**2, idx


def _assert_matches_exact_knn(adata, coords, k):
    idx = adata.obsm["index_neighbor"]
    dist = adata.obsm["distance_neighbor"]
    true_dist, true_idx = _exact_squared_knn(coords, k)
    assert idx.shape == dist.shape == (len(coords), k)
    assert not (idx == np.arange(len(coords))[:, None]).any()
    assert np.all(np.diff(dist, axis=1) >= 0)
    np.testing.assert_allclose(dist, true_dist, rtol=1e-4, atol=1e-4)
    # Cells 0 and 1 coincide: each must be the other's nearest neighbor at distance 0.
    assert idx[0, 0] == 1
    assert idx[1, 0] == 0
    overlap = np.mean([len(set(idx[i]) & set(true_idx[i])) / k for i in range(len(coords))])
    assert overlap > 0.99  # only exact distance ties at the k-th neighbor may differ


def test_compute_neighbors_squidpy_squared_sorted_no_self():
    """Regression test for scverse/scvi-tools#3977 (mixin path): squidpy backend stores the
    true kNN with squared Euclidean distances, sorted, self excluded."""
    pytest.importorskip("squidpy")
    coords = _coords_with_coincident_pair()
    adata = _make_coords_adata(n=len(coords))
    adata.obsm["spatial"] = coords

    SpatialNeighborhoodMixin.compute_neighbors(adata, n_neighs=8, backend="squidpy")

    _assert_matches_exact_knn(adata, coords, 8)
    config = adata.uns["_spatial_compute_neighbors_config"]
    assert config["_version"] >= 2
    assert config["n_neighs"] == 8


def test_compute_neighbors_rapids_drops_self_by_index(monkeypatch):
    """RAPIDS backend drops self by index, not by assuming it is column 0 (it isn't when
    cells share a centroid). cuML/CuPy are faked with sklearn/numpy."""
    import sys
    import types

    from sklearn.neighbors import NearestNeighbors

    coords = _coords_with_coincident_pair()

    class _FakeNN:
        def __init__(self, n_neighbors):
            self._nn = NearestNeighbors(n_neighbors=n_neighbors)

        def fit(self, x):
            self._nn.fit(x)

        def kneighbors(self, x):
            dist, idx = self._nn.kneighbors(x)
            # Emulate cuML returning the coincident neighbor before self.
            idx[[0, 1], :2] = [[1, 0], [0, 1]]
            return dist, idx

    cuml = types.ModuleType("cuml")
    cuml.neighbors = types.SimpleNamespace(NearestNeighbors=_FakeNN)
    cupy = types.ModuleType("cupy")
    cupy.asarray = np.asarray
    cupy.asnumpy = np.asarray
    monkeypatch.setitem(sys.modules, "cuml", cuml)
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    adata = _make_coords_adata(n=len(coords))
    adata.obsm["spatial"] = coords
    SpatialNeighborhoodMixin.compute_neighbors(adata, n_neighs=8, backend="rapids")

    _assert_matches_exact_knn(adata, coords, 8)
    assert adata.uns["_spatial_compute_neighbors_config"]["backend"] == "rapids"
