import json

import anndata
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import TableModel

from scviva.tools.csde._annotation import load_annotations, prepare_csde_inputs


@pytest.fixture
def annotation_dir_and_sdata(tmp_path):
    n = 40
    n_genes = 8
    rng = np.random.default_rng(0)

    obs = pd.DataFrame(
        {
            "cell_type": rng.choice(["macrophage", "tcell", "other"], size=n),
            "spatial_group": rng.choice([0, 1], size=n),
            "region": "cells",
            "instance_id": np.arange(n),
        }
    )
    X = rng.poisson(3.0, size=(n, n_genes)).astype(float)
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

    # 10 cells are "annotated" (manually validated)
    annotated_ids = list(adata.obs_names[:10])
    metadata = pd.DataFrame(
        {
            "cell_id": annotated_ids,
            "sampling_weight": rng.uniform(0.5, 2.0, size=10),
        }
    )
    annotations = {cid: bool(rng.integers(0, 2)) for cid in annotated_ids}
    # force at least one True and one False so is_correct has both values
    annotations[annotated_ids[0]] = True
    annotations[annotated_ids[1]] = False

    config = {"cell_type_key": "cell_type", "cell_type_of_interest": "macrophage"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    return tmp_path, sdata


def test_load_annotations_merges_metadata_and_annotations(annotation_dir_and_sdata):
    annotation_dir, _sdata = annotation_dir_and_sdata
    df = load_annotations(annotation_dir)
    assert len(df) == 10
    assert "is_correct" in df.columns
    assert df["is_correct"].dtype == bool


def test_load_annotations_missing_file_raises(tmp_path):
    (tmp_path / "metadata.csv").write_text("cell_id,sampling_weight\ncell_0,1.0\n")
    with pytest.raises(FileNotFoundError):
        load_annotations(tmp_path)


def test_prepare_csde_inputs_splits_gt_and_other(annotation_dir_and_sdata):
    annotation_dir, sdata = annotation_dir_and_sdata
    result = prepare_csde_inputs(annotation_dir, sdata=sdata, n_cells_expressed_threshold=0)
    adata_gt, adata_other = result["adata_gt"], result["adata_other"]

    assert len(adata_gt) == 10
    assert len(adata_other) == 30
    for col in ["prediction", "annotation", "is_correct", "sampling_weight"]:
        assert col in adata_gt.obs.columns
    assert "prediction" in adata_other.obs.columns
    assert set(adata_gt.var_names) == set(adata_other.var_names)


def test_prepare_csde_inputs_gene_filter(annotation_dir_and_sdata):
    annotation_dir, sdata = annotation_dir_and_sdata
    result_strict = prepare_csde_inputs(
        annotation_dir, sdata=sdata, n_cells_expressed_threshold=1000
    )
    assert result_strict["adata_gt"].shape[1] == 0
