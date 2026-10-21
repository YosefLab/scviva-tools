"""Tests for DestVI model."""

import sys
import types

import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid
from scvi.model import CondSCVI

from scviva.model._destvi import DestVI


@pytest.fixture(scope="session")
def destvi_data():
    sc_adata = synthetic_iid(n_labels=4, n_genes=50)
    sc_adata.layers["counts"] = sc_adata.X.copy()
    st_adata = synthetic_iid(n_labels=4, n_genes=50, n_batches=1)
    st_adata.layers["counts"] = st_adata.X.copy()
    st_adata.obsm["spatial"] = np.random.rand(st_adata.n_obs, 2)
    return sc_adata, st_adata


def test_destvi_from_rna_model_train(destvi_data):
    sc_adata, st_adata = destvi_data
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=1)
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=1)
    assert st_model.is_trained


def test_destvi_get_proportions_df(destvi_data):
    sc_adata, st_adata = destvi_data
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=1)
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=1)
    df = st_model.get_proportions_df(st_adata)
    assert df.shape[1] == 4  # n_labels
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-4)


def test_destvi_get_latent_dist_rapids_preserves_tuple(destvi_data, monkeypatch):
    sc_adata, st_adata = destvi_data
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=1)
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model, n_latent_amortization=10)
    st_model.train(max_epochs=1)

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=np.asarray))

    latent = st_model.get_latent_representation(return_dist=True, backend="rapids")

    assert isinstance(latent, tuple)
    assert len(latent) == 2
    assert latent[0].shape[0] == st_adata.n_obs
    assert latent[1].shape == latent[0].shape


def test_condscvi_not_re_exported():
    """CondSCVI must NOT be exported from scviva — users import from scvi.model directly."""
    import scviva

    with pytest.raises(AttributeError):
        _ = scviva.CondSCVI


def test_destvi_validation():
    # DestVI must be trainable with a validation set: `early_stopping`,
    # `check_val_every_n_epoch` and `train_size < 1.0` previously crashed in the loss because the
    # augmentation branch was not gated on `self.training` (validation produces no augmentation
    # tensors). See `MRDeconv.loss`.
    n_latent = 2
    n_labels = 5
    dataset = synthetic_iid(n_labels=n_labels)
    CondSCVI.setup_anndata(dataset, labels_key="labels", batch_key="batch")
    sc_model = CondSCVI(dataset, n_latent=n_latent, n_layers=2, prior="mog", num_classes_mog=10)
    sc_model.train(1, train_size=0.9)

    DestVI.setup_anndata(dataset, layer=None)

    # train_size < 1.0 with validation evaluated every epoch
    model = DestVI.from_rna_model(dataset, sc_model, amortization="both")
    model.train(max_epochs=2, train_size=0.9, check_val_every_n_epoch=1)
    assert "validation_loss" in model.history
    assert not np.isnan(model.history["validation_loss"].values[0][0])

    # early stopping (requires a working validation pass)
    model = DestVI.from_rna_model(dataset, sc_model, amortization="both")
    model.train(max_epochs=2, train_size=0.9, early_stopping=True)
    assert "validation_loss" in model.history

    # Non-amortized spot parameters (here `V`, since amortization="latent") are only trained on the
    # training spots, so a validation set would be evaluated on randomly-initialized parameters.
    # Reject it. ("proportion"/"none" are not exercised here because mog priors require amortized
    # latents, but they hit the same `amortization != 'both'` guard.)
    model = DestVI.from_rna_model(dataset, sc_model, amortization="latent")
    with pytest.raises(ValueError, match="amortization='both'"):
        model.train(max_epochs=2, train_size=0.9)
    with pytest.raises(ValueError, match="amortization='both'"):
        model.train(max_epochs=2, early_stopping=True)
    # train_size=1.0 (all spots trained) must still work
    model.train(max_epochs=1)


@pytest.mark.parametrize("prior", [None, "normal"])
def test_destvi_vamp_prior_from_normal_condscvi(prior):
    # CondSCVI with its default prior: the VampPrior comes from get_vamp_prior (numpy arrays)
    n_labels = 3
    dataset = synthetic_iid(n_labels=n_labels)
    CondSCVI.setup_anndata(dataset, labels_key="labels")
    kwargs = {} if prior is None else {"prior": prior}
    sc_model = CondSCVI(dataset, n_latent=2, **kwargs)
    sc_model.train(1, train_size=1)

    DestVI.setup_anndata(dataset, layer=None)
    spatial_model = DestVI.from_rna_model(dataset, sc_model)
    spatial_model.train(max_epochs=1)
    assert not np.isnan(spatial_model.history["elbo_train"].values[0][0])
    assert spatial_model.get_proportions().shape == (dataset.n_obs, n_labels)


def test_destvi_qz_prior_follows_module():
    # the VampPrior is rebuilt from the buffers, so .to() moves it with the module
    dataset = synthetic_iid(n_labels=3)
    CondSCVI.setup_anndata(dataset, labels_key="labels")
    sc_model = CondSCVI(dataset, n_latent=2, prior="normal")
    sc_model.train(1, train_size=1)
    DestVI.setup_anndata(dataset, layer=None)
    module = DestVI.from_rna_model(dataset, sc_model).module
    assert module.qz_prior.mean.dtype == module.mean_vprior.dtype == torch.float32
    module.double()
    assert module.qz_prior.mean.dtype == torch.float64
    assert module.qz_prior.sample([2]).dtype == torch.float64
