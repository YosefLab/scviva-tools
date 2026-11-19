"""Tests for GIMVI model."""

from __future__ import annotations

import numpy as np
import pytest
from scvi.data import synthetic_iid

from scviva.model._gimvi import GIMVI


@pytest.fixture(scope="module")
def gimvi_data():
    """Shared seq + spatial synthetic datasets."""
    np.random.seed(0)
    adata_seq = synthetic_iid(n_genes=50, n_batches=2, sparse_format=None)
    adata_seq.layers["counts"] = adata_seq.X.copy()

    adata_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    adata_spatial.layers["counts"] = adata_spatial.X.copy()
    # Make spatial genes a strict subset of seq genes
    adata_spatial.var_names = adata_seq.var_names[:20]

    return adata_seq, adata_spatial


def _setup_and_build(adata_seq, adata_spatial, **model_kwargs):
    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    GIMVI.setup_anndata(adata_spatial, layer="counts")
    return GIMVI(adata_seq, adata_spatial, **model_kwargs)


def test_gimvi_train(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)
    assert model.is_trained


def test_gimvi_get_latent_representation(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    latents = model.get_latent_representation()
    assert isinstance(latents, list)
    assert len(latents) == 2
    assert latents[0].shape[0] == adata_seq.n_obs
    assert latents[1].shape[0] == adata_spatial.n_obs
    assert latents[0].shape[1] == latents[1].shape[1]  # same latent dim


def test_gimvi_get_latent_rapids_stub(gimvi_data, monkeypatch):
    """Verify rapids path converts to cupy arrays (stubbed)."""
    import sys
    import types

    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=np.asarray))
    latents = model.get_latent_representation(backend="rapids")
    assert len(latents) == 2


def test_gimvi_get_imputed_values(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    imputed = model.get_imputed_values()
    assert isinstance(imputed, list)
    assert len(imputed) == 2
    assert imputed[0].shape[0] == adata_seq.n_obs
    assert imputed[1].shape[0] == adata_spatial.n_obs


def test_gimvi_same_adata_raises(gimvi_data):
    adata_seq, _ = gimvi_data
    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    with pytest.raises(ValueError, match="cannot point to the same object"):
        GIMVI(adata_seq, adata_seq)


def test_gimvi_spatial_genes_not_subset_raises(gimvi_data):
    adata_seq, _ = gimvi_data
    np.random.seed(1)
    bad_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    bad_spatial.layers["counts"] = bad_spatial.X.copy()
    # Force var_names that are NOT in adata_seq
    bad_spatial.var_names = [f"nonexistent_gene_{i}" for i in range(20)]

    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    GIMVI.setup_anndata(bad_spatial, layer="counts")
    with pytest.raises(ValueError, match="subset"):
        GIMVI(adata_seq, bad_spatial)


def test_gimvi_setup_spatialdata_inherits():
    """setup_spatialdata classmethod must be present (inherited from SpatialBaseModel)."""
    assert hasattr(GIMVI, "setup_spatialdata")
    assert hasattr(GIMVI, "from_spatialdata")


def test_gimvi_gene_label_dispersion():
    """dispersion='gene-label' must not crash (y.type was a method object bug)."""
    np.random.seed(0)
    adata_seq = synthetic_iid(n_genes=50, n_batches=2, n_labels=3, sparse_format=None)
    adata_seq.layers["counts"] = adata_seq.X.copy()
    adata_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    adata_spatial.layers["counts"] = adata_spatial.X.copy()
    adata_spatial.var_names = adata_seq.var_names[:20]

    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch", labels_key="labels")
    GIMVI.setup_anndata(adata_spatial, layer="counts")
    n_labels = adata_seq.obs["labels"].nunique()
    model = GIMVI(adata_seq, adata_spatial, dispersion="gene-label", n_labels=n_labels)
    model.train(max_epochs=1)
    assert model.is_trained


def test_gimvi_lazy_import():
    """GIMVI must be accessible via scviva top-level lazy import."""
    import scviva

    assert hasattr(scviva, "GIMVI") or "GIMVI" in scviva.__all__


def test_gimvi_save_forwards_anndata_write_kwargs(gimvi_data, tmp_path, monkeypatch):
    """`**anndata_write_kwargs` must reach AnnData.write for both datasets."""
    from anndata import AnnData

    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq.copy(), adata_spatial.copy())

    calls = []
    original_write = AnnData.write

    def spy_write(self, filename=None, **kwargs):
        calls.append(kwargs)
        return original_write(self, filename, **kwargs)

    monkeypatch.setattr(AnnData, "write", spy_write)
    model.save(tmp_path / "gimvi", save_anndata=True, compression="gzip")

    assert len(calls) == 2
    assert all(kw.get("compression") == "gzip" for kw in calls)
