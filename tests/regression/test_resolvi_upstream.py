"""Regression tests: scviva.ResolVI vs scvi.external.resolvi.ResolVI.

Each test runs the exact same operations on the same data with the same random seed
on both implementations and asserts outputs are identical (within float tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest
import scvi as scvi_pkg
import torch
from scvi.data import synthetic_iid
from scvi.external.resolvi import RESOLVI as ScviResolVI

from scviva.model import ResolVI as SpatialResolVI

SEED = 42
N_EPOCHS = 1


@pytest.fixture(scope="module")
def adata():
    np.random.seed(SEED)
    adata = synthetic_iid(
        generate_coordinates=True,
        n_regions=5,
        n_proteins=10,
    )
    adata.obsm["X_spatial"] = adata.obsm["coordinates"]
    adata.obs["cell_area"] = np.random.gamma(2.0, 1.0, size=adata.n_obs)
    return adata


def _train_scvi(adata, seed=SEED, **model_kwargs):
    """Train scvi.external.resolvi.ResolVI with a fixed seed."""
    scvi_pkg.settings.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    ScviResolVI.setup_anndata(adata)
    model = ScviResolVI(adata, **model_kwargs)
    model.train(max_epochs=N_EPOCHS, lr=1e-3)
    return model


def _train_scviva(adata, seed=SEED, **model_kwargs):
    """Train scviva.ResolVI with a fixed seed."""
    scvi_pkg.settings.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Reuse the neighbors scvi-tools just computed: scviva's ``_prepare_data`` intentionally
    # differs from upstream since the scverse/scvi-tools#3977 fix, and these tests compare
    # model math on identical inputs.
    SpatialResolVI.setup_anndata(adata, prepare_data=False)
    model = SpatialResolVI(adata, **model_kwargs)
    model.train(max_epochs=N_EPOCHS, lr=1e-3)
    return model


def test_resolvi_latent_representation_matches(adata):
    """Latent representations from both implementations must match."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_scviva(adata)

    latent_scvi = scvi_model.get_latent_representation()
    latent_spatial = spatial_model.get_latent_representation()

    assert latent_scvi.shape == latent_spatial.shape, (
        f"Shape mismatch: scvi={latent_scvi.shape}, scviva={latent_spatial.shape}"
    )
    # Outputs must be numerically identical (same model, same seed, same data)
    np.testing.assert_allclose(
        latent_scvi,
        latent_spatial,
        atol=1e-5,
        err_msg="Latent representations differ between scvi and scviva ResolVI",
    )


def test_resolvi_elbo_matches(adata):
    """Training ELBO history must match between implementations."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_scviva(adata)

    elbo_scvi = np.array(scvi_model.history_["elbo_train"].values, dtype=float)
    elbo_spatial = np.array(spatial_model.history_["elbo_train"].values, dtype=float)

    assert len(elbo_scvi) == len(elbo_spatial)
    np.testing.assert_allclose(
        elbo_scvi,
        elbo_spatial,
        atol=1e-4,
        err_msg="Training ELBO differs between scvi and scviva ResolVI",
    )


def test_resolvi_normalized_expression_matches(adata):
    """Normalized expression must match between implementations."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_scviva(adata)

    torch.manual_seed(SEED)
    expr_scvi = scvi_model.get_normalized_expression(n_samples=5, library_size=10000)
    torch.manual_seed(SEED)
    expr_spatial = spatial_model.get_normalized_expression(n_samples=5, library_size=10000)

    assert expr_scvi.shape == expr_spatial.shape
    np.testing.assert_allclose(
        expr_scvi.values,
        expr_spatial.values,
        atol=1e-4,
        err_msg="Normalized expression differs between scvi and scviva ResolVI",
    )


def test_resolvi_save_load_matches(adata, tmp_path):
    """Save/load round-trip must produce identical outputs in both implementations."""
    spatial_model = _train_scviva(adata)
    latent_before = spatial_model.get_latent_representation()

    save_path = str(tmp_path / "resolvi_model")
    spatial_model.save(save_path, save_anndata=True, overwrite=True)
    loaded = SpatialResolVI.load(save_path)
    latent_after = loaded.get_latent_representation()

    np.testing.assert_array_equal(
        latent_before,
        latent_after,
        err_msg="Latent representation changed after save/load",
    )


def test_resolvi_differential_expression_runs(adata):
    """Differential expression must complete without error in both implementations."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_scviva(adata)

    de_scvi = scvi_model.differential_expression(groupby="labels")
    de_spatial = spatial_model.differential_expression(groupby="labels")

    assert de_scvi.shape == de_spatial.shape, (
        f"DE output shape mismatch: scvi={de_scvi.shape}, scviva={de_spatial.shape}"
    )
    assert list(de_scvi.columns) == list(de_spatial.columns), (
        "DE column names differ between implementations"
    )


@pytest.mark.parametrize("weights", ["importance", "uniform"])
def test_resolvi_differential_expression_weights(adata, weights):
    """DE with both weight types must produce same-shaped results in both implementations."""
    if weights == "uniform":
        scvi_model = _train_scvi(adata)
        spatial_model = _train_scviva(adata)
        de_scvi = scvi_model.differential_expression(groupby="labels", weights=weights)
        de_spatial = spatial_model.differential_expression(groupby="labels", weights=weights)
    else:
        scvi_model = _train_scvi(adata)
        spatial_model = _train_scviva(adata)
        de_scvi = scvi_model.differential_expression(groupby="labels", weights=weights)
        de_spatial = spatial_model.differential_expression(groupby="labels", weights=weights)

    assert de_scvi.shape == de_spatial.shape


def test_resolvi_size_factor_matches(adata):
    """Size-factor-scaled models must produce matching latent representations."""
    scvi_pkg.settings.seed = SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ScviResolVI.setup_anndata(adata, size_factor_key="cell_area")
    scvi_model = ScviResolVI(adata, size_scaling=True)
    scvi_model.train(max_epochs=N_EPOCHS)

    scvi_pkg.settings.seed = SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Reuse the neighbors scvi-tools just computed: scviva's ``_prepare_data`` intentionally
    # differs from upstream since the scverse/scvi-tools#3977 fix, and these tests compare
    # model math on identical inputs.
    SpatialResolVI.setup_anndata(adata, size_factor_key="cell_area", prepare_data=False)
    spatial_model = SpatialResolVI(adata, size_scaling=True)
    spatial_model.train(max_epochs=N_EPOCHS)

    latent_scvi = scvi_model.get_latent_representation()
    latent_spatial = spatial_model.get_latent_representation()

    assert latent_scvi.shape == latent_spatial.shape
    np.testing.assert_allclose(latent_scvi, latent_spatial, atol=1e-5)
