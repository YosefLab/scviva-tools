import anndata
import numpy as np
import pandas as pd
import pytest

from scviva.tools.csde.tools import run_csde


@pytest.fixture
def csde_synthetic_adata():
    np.random.seed(0)
    n_genes = 10

    n_pred = 100
    X_pred = np.random.poisson(lam=2.0, size=(n_pred, n_genes)).astype(float)
    obs_pred = pd.DataFrame(
        {"cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_pred)}
    )
    adata_pred = anndata.AnnData(X=X_pred, obs=obs_pred)
    adata_pred.var_names = [f"Gene_{i}" for i in range(n_genes)]

    n_gt = 50
    X_gt = np.random.poisson(lam=2.0, size=(n_gt, n_genes)).astype(float)
    obs_gt = pd.DataFrame(
        {
            "cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_gt),
            "is_correct": np.random.choice([True, False], size=n_gt),
        }
    )
    adata_gt = anndata.AnnData(X=X_gt, obs=obs_gt)
    adata_gt.var_names = [f"Gene_{i}" for i in range(n_genes)]

    # guarantee both populations are present
    adata_pred.obs.iloc[0, 0] = "TypeA"
    adata_pred.obs.iloc[1, 0] = "TypeB"
    adata_gt.obs.iloc[0, 0] = "TypeA"
    adata_gt.obs.iloc[1, 0] = "TypeB"
    return adata_pred, adata_gt


def test_run_csde(csde_synthetic_adata):
    adata_pred, adata_gt = csde_synthetic_adata
    res = run_csde(
        adata_pred=adata_pred,
        adata_gt=adata_gt,
        pred_cell_pop_key="cell_type",
        cell_pop_a="TypeA",
        cell_pop_b="TypeB",
        gt_key="is_correct",
        optimizer="gd",
        optimizer_kwargs={"n_iter": 10},
    )
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert list(res.columns) == ["log_fold_change", "p_value", "p_value_adj"]
    assert not res.isnull().values.any()


@pytest.mark.parametrize("noise_model", ["poisson", "nb"])
def test_run_csde_with_importance_weights(csde_synthetic_adata, noise_model):
    adata_pred, adata_gt = csde_synthetic_adata
    rng = np.random.default_rng(0)
    importance_weights = rng.uniform(0.5, 2.0, size=len(adata_gt))

    res = run_csde(
        adata_pred=adata_pred,
        adata_gt=adata_gt,
        pred_cell_pop_key="cell_type",
        cell_pop_a="TypeA",
        cell_pop_b="TypeB",
        gt_key="is_correct",
        optimizer="gd",
        optimizer_kwargs={"n_iter": 10},
        importance_weights=importance_weights,
        noise_model=noise_model,
    )
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 10
    assert not res.isnull().values.any()


def test_importance_weights_wrong_shape_raises(csde_synthetic_adata):
    from scviva.tools.csde._model_poisson import PoissonIntercept

    adata_pred, adata_gt = csde_synthetic_adata
    x_gt, y_gt = adata_gt.X.astype(float), np.zeros(len(adata_gt), dtype=int)
    x_unl = adata_pred.X.astype(float)
    y_hat = np.zeros(len(adata_gt), dtype=int)
    y_unl = np.zeros(len(adata_pred), dtype=int)

    bad_weights = np.ones(len(adata_gt) + 5)
    with pytest.raises(ValueError):
        PoissonIntercept(
            inputs_gt=(x_gt, y_gt),
            inputs_hat=(x_gt, y_hat),
            inputs_unl=(x_unl, y_unl),
            importance_weights=bad_weights,
        )


def test_run_csde_unknown_noise_model_raises(csde_synthetic_adata):
    adata_pred, adata_gt = csde_synthetic_adata
    with pytest.raises(ValueError):
        run_csde(
            adata_pred=adata_pred,
            adata_gt=adata_gt,
            pred_cell_pop_key="cell_type",
            cell_pop_a="TypeA",
            cell_pop_b="TypeB",
            gt_key="is_correct",
            noise_model="bogus",
        )


def test_run_csde_unknown_cell_pop_a_raises(csde_synthetic_adata):
    adata_pred, adata_gt = csde_synthetic_adata
    with pytest.raises(ValueError):
        run_csde(
            adata_pred=adata_pred,
            adata_gt=adata_gt,
            pred_cell_pop_key="cell_type",
            cell_pop_a="NonexistentType",
            cell_pop_b="TypeB",
            gt_key="is_correct",
        )


def test_run_csde_unknown_cell_pop_b_raises(csde_synthetic_adata):
    adata_pred, adata_gt = csde_synthetic_adata
    with pytest.raises(ValueError):
        run_csde(
            adata_pred=adata_pred,
            adata_gt=adata_gt,
            pred_cell_pop_key="cell_type",
            cell_pop_a="TypeA",
            cell_pop_b="NonexistentType",
            gt_key="is_correct",
        )


def test_run_csde_reorders_mismatched_gene_order(csde_synthetic_adata):
    """A shuffled but identical gene set in adata_pred must not change results."""
    adata_pred, adata_gt = csde_synthetic_adata
    shuffled_genes = list(adata_pred.var_names[::-1])
    adata_pred_shuffled = adata_pred[:, shuffled_genes].copy()

    kwargs = {
        "pred_cell_pop_key": "cell_type",
        "cell_pop_a": "TypeA",
        "cell_pop_b": "TypeB",
        "gt_key": "is_correct",
        "optimizer": "gd",
        "optimizer_kwargs": {"n_iter": 10},
    }
    res_aligned = run_csde(adata_pred=adata_pred, adata_gt=adata_gt, **kwargs)
    res_shuffled = run_csde(adata_pred=adata_pred_shuffled, adata_gt=adata_gt, **kwargs)

    pd.testing.assert_frame_equal(res_aligned, res_shuffled.reindex(res_aligned.index))


def test_run_csde_mismatched_gene_sets_raises(csde_synthetic_adata):
    adata_pred, adata_gt = csde_synthetic_adata
    renamed_genes = list(adata_pred.var_names[:-1]) + ["Extra_gene"]
    adata_pred_bad = adata_pred.copy()
    adata_pred_bad.var_names = renamed_genes

    with pytest.raises(ValueError, match="same genes"):
        run_csde(
            adata_pred=adata_pred_bad,
            adata_gt=adata_gt,
            pred_cell_pop_key="cell_type",
            cell_pop_a="TypeA",
            cell_pop_b="TypeB",
            gt_key="is_correct",
        )
