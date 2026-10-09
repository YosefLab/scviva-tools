from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData


def _make_amici_adata(n_obs: int = 24, n_genes: int = 12, n_labels: int = 3) -> AnnData:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_genes)).astype(np.float32)
    adata = AnnData(X=x)
    adata.layers["expr"] = x.copy()
    adata.obs["cell_type"] = [f"ct_{i % n_labels}" for i in range(n_obs)]
    adata.obsm["spatial"] = rng.normal(size=(n_obs, 2)).astype(np.float32)
    return adata


def test_amici_setup_anndata_registers_neighbors():
    from scviva.external.amici import AMICI
    from scviva.external.amici._constants import AMICI_REGISTRY_KEYS

    adata = _make_amici_adata()
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=4,
    )

    assert AMICI_REGISTRY_KEYS.NN_IDX_KEY in adata.obsm
    assert AMICI_REGISTRY_KEYS.NN_DIST_KEY in adata.obsm
    assert adata.obsm[AMICI_REGISTRY_KEYS.NN_IDX_KEY].shape == (adata.n_obs, 4)
    assert adata.obsm[AMICI_REGISTRY_KEYS.NN_DIST_KEY].shape == (adata.n_obs, 4)


def test_amici_train_and_predict_shape():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=3,
    )
    model = AMICI(
        adata,
        n_label_embed=4,
        n_nn_embed=8,
        n_hidden=16,
    )

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    predictions = model.get_predictions(batch_size=8)

    assert model.is_trained
    assert predictions.shape == (adata.n_obs, adata.n_vars)
    assert not np.any(np.isnan(predictions))


def test_amici_get_predictions_returns_residuals():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=3,
    )
    model = AMICI(adata, n_label_embed=4, n_nn_embed=8, n_hidden=16)

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    residuals = model.get_predictions(batch_size=8, get_residuals=True)

    assert residuals.shape == (adata.n_obs, adata.n_vars)
    assert not np.any(np.isnan(residuals))


def test_amici_get_predictions_can_store_in_anndata():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=3,
    )
    model = AMICI(adata, n_label_embed=4, n_nn_embed=8, n_hidden=16)

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    predictions = model.get_predictions(batch_size=8, store_key="amici_prediction")

    assert "amici_prediction" in adata.obsm
    np.testing.assert_allclose(adata.obsm["amici_prediction"], predictions)


def test_amici_get_attention_patterns_shape_and_store():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    n_neighbors = 3
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=n_neighbors,
    )
    model = AMICI(adata, n_label_embed=4, n_nn_embed=8, n_hidden=16)

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    attention = model.get_attention_patterns(batch_size=8, store_key="amici_attention")

    assert attention.shape == (adata.n_obs, n_neighbors)
    assert "amici_attention" in adata.obsm
    np.testing.assert_allclose(adata.obsm["amici_attention"], attention)
    np.testing.assert_allclose(attention.sum(axis=1), 1.0, atol=1e-5)


def test_amici_get_nn_embed_shape_and_store():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    n_neighbors = 3
    n_nn_embed = 8
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=n_neighbors,
    )
    model = AMICI(adata, n_label_embed=4, n_nn_embed=n_nn_embed, n_hidden=16)

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    nn_embed = model.get_nn_embed(batch_size=8, store_key="X_amici_nn")

    assert nn_embed.shape == (adata.n_obs, n_neighbors, n_nn_embed)
    assert "X_amici_nn" in adata.obsm
    np.testing.assert_allclose(adata.obsm["X_amici_nn"], nn_embed)


def test_amici_setup_anndata_allows_same_label_neighbors_when_requested():
    from scviva.external.amici import AMICI
    from scviva.external.amici._constants import AMICI_REGISTRY_KEYS

    adata = _make_amici_adata(n_obs=8, n_labels=1)
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=2,
        exclude_self_labels=False,
    )

    nn_idx = adata.obsm[AMICI_REGISTRY_KEYS.NN_IDX_KEY]
    assert nn_idx.shape == (adata.n_obs, 2)
    assert not np.any(nn_idx == np.arange(adata.n_obs)[:, None])


def test_amici_external_import():
    from scviva.external import AMICI
    from scviva.external.amici import AMICI as SubmoduleAMICI
    from scviva.external.amici import AMICIModule

    assert AMICI is SubmoduleAMICI
    assert AMICIModule is not None


def test_amici_setup_anndata_requires_labels_for_label_exclusion():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    with pytest.raises(ValueError, match="labels_key"):
        AMICI.setup_anndata(
            adata,
            layer="expr",
            labels_key=None,
            spatial_key="spatial",
            exclude_self_labels=True,
        )
