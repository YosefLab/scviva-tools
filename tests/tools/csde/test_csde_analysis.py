import json

import anndata
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import TableModel

from scviva.tools.csde import CSDEAnalysis


@pytest.fixture
def annotation_dir_and_sdata(tmp_path):
    n = 60
    n_genes = 6
    rng = np.random.default_rng(2)

    true_effect = rng.uniform(-0.4, 0.4, n_genes)
    cell_type = rng.choice(["macrophage", "other"], size=n, p=[0.5, 0.5])
    spatial_group = rng.choice([0, 1], size=n)
    is_coi = cell_type == "macrophage"
    rates = np.exp(1.0 + np.outer(is_coi & (spatial_group == 1), true_effect))
    X = rng.poisson(rates).astype(float)

    obs = pd.DataFrame(
        {
            "cell_type": cell_type,
            "spatial_group": spatial_group,
            "region": "cells",
            "instance_id": np.arange(n),
        }
    )
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata = TableModel.parse(
        adata, region="cells", region_key="region", instance_key="instance_id"
    )
    # No shapes/images/points elements registered: prepare_csde_inputs only ever reads
    # sdata["table"], so they're unnecessary here. SpatialData emits a benign UserWarning
    # ("table is annotating 'cells', which is not present...") for the missing region --
    # verified harmless during planning, safe to ignore in this fixture.
    sdata = sd.SpatialData(tables={"table": adata})

    annotated_ids = list(adata.obs_names[:20])
    metadata = pd.DataFrame(
        {"cell_id": annotated_ids, "sampling_weight": rng.uniform(0.5, 2.0, size=20)}
    )
    annotations = dict.fromkeys(annotated_ids, True)  # all validated as correct

    config = {"cell_type_key": "cell_type", "cell_type_of_interest": "macrophage"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    return tmp_path, sdata


def test_from_spatialdata_builds_analysis(annotation_dir_and_sdata):
    annotation_dir, sdata = annotation_dir_and_sdata
    analysis = CSDEAnalysis.from_spatialdata(
        sdata, annotation_dir=annotation_dir, n_cells_expressed_threshold=0
    )
    assert isinstance(analysis, CSDEAnalysis)
    assert analysis.pred_cell_pop_key == "prediction"
    assert analysis.cell_pop_a == 0
    assert analysis.cell_pop_b == 1
    assert analysis.gt_key == "is_correct"
    assert analysis.importance_weights is not None
    assert len(analysis.importance_weights) == len(analysis.adata_gt)


def test_fit_then_test_differential_expression(annotation_dir_and_sdata):
    annotation_dir, sdata = annotation_dir_and_sdata
    analysis = CSDEAnalysis.from_spatialdata(
        sdata, annotation_dir=annotation_dir, n_cells_expressed_threshold=0
    )
    returned = analysis.fit(noise_model="poisson", optimizer="gd", optimizer_kwargs={"n_iter": 20})
    assert returned is analysis  # fit() returns self

    res = analysis.test_differential_expression()
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ["log_fold_change", "p_value", "p_value_adj"]
    assert len(res) == analysis.adata_gt.shape[1]
    assert not res.isnull().values.any()


def test_fit_with_default_optimizer_does_not_raise(annotation_dir_and_sdata):
    """Regression test: CSDEAnalysis.fit()'s default optimizer value must be a value
    InterceptPPIModel actually accepts ("gd" or "lbfgs"), not "adam" (a different,
    inner-level parameter). A prior version defaulted to "adam" and crashed with
    ValueError: Unknown optimizer: adam on any call to .fit() with no arguments.
    """
    annotation_dir, sdata = annotation_dir_and_sdata
    analysis = CSDEAnalysis.from_spatialdata(
        sdata, annotation_dir=annotation_dir, n_cells_expressed_threshold=0
    )
    # Deliberately call fit() with NO optimizer override, to exercise the real default.
    # Keep iteration count low via optimizer_kwargs so the test runs fast.
    analysis.fit(optimizer_kwargs={"n_iter": 20})
    res = analysis.test_differential_expression()
    assert not res.isnull().values.any()


def test_test_differential_expression_before_fit_raises(annotation_dir_and_sdata):
    annotation_dir, sdata = annotation_dir_and_sdata
    analysis = CSDEAnalysis.from_spatialdata(
        sdata, annotation_dir=annotation_dir, n_cells_expressed_threshold=0
    )
    with pytest.raises(RuntimeError):
        analysis.test_differential_expression()


def test_base_constructor_works_with_plain_anndata_no_spatialdata():
    """CSDEAnalysis composes with any upstream scviva-tools output written into plain
    AnnData .obs/.layers (e.g. a scANVI cell_type column) -- from_spatialdata is one
    entry point, not the only one.
    """
    rng = np.random.default_rng(3)
    n_genes = 5
    obs_pred = pd.DataFrame({"predicted_cell_type": rng.choice(["A", "B"], size=40)})
    adata_pred = anndata.AnnData(
        X=rng.poisson(2.0, size=(40, n_genes)).astype(float), obs=obs_pred
    )
    adata_pred.var_names = [f"g{i}" for i in range(n_genes)]

    obs_gt = pd.DataFrame(
        {
            "predicted_cell_type": rng.choice(["A", "B"], size=15),
            "validated": rng.choice([True, False], size=15),
        }
    )
    adata_gt = anndata.AnnData(X=rng.poisson(2.0, size=(15, n_genes)).astype(float), obs=obs_gt)
    adata_gt.var_names = adata_pred.var_names
    adata_pred.obs.iloc[0, 0] = "A"
    adata_pred.obs.iloc[1, 0] = "B"
    adata_gt.obs.iloc[0, 0] = "A"
    adata_gt.obs.iloc[1, 0] = "B"

    analysis = CSDEAnalysis(
        adata_pred=adata_pred,
        adata_gt=adata_gt,
        pred_cell_pop_key="predicted_cell_type",
        cell_pop_a="A",
        cell_pop_b="B",
        gt_key="validated",
    )
    analysis.fit(noise_model="poisson", optimizer="gd", optimizer_kwargs={"n_iter": 10})
    res = analysis.test_differential_expression()
    assert not res.isnull().values.any()
