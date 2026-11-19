from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_annotations(annotation_dir: str | Path) -> pd.DataFrame:
    """Merge `metadata.csv` and `annotations.json` into one DataFrame of annotated cells.

    Input is upstream CSDE's `export.py`/`annotate.py` output; adds a boolean `is_correct`
    column.
    """
    annotation_dir = Path(annotation_dir)
    metadata = pd.read_csv(annotation_dir / "metadata.csv")
    metadata["cell_id"] = metadata["cell_id"].astype(str)

    ann_path = annotation_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"No annotations found at {ann_path}. Run upstream CSDE's scripts/annotate.py first "
            "(see https://github.com/YosefLab/CSDE)."
        )
    with open(ann_path) as f:
        annotations = json.load(f)

    metadata["is_correct"] = metadata["cell_id"].map(annotations)
    return metadata[metadata["is_correct"].notna()].copy()


def prepare_csde_inputs(
    annotation_dir: str | Path,
    sdata=None,
    spatial_group_key: str = "spatial_group",
    spatial_group_target: int = 1,
    spatial_group_reference: int = 0,
    layer: str | None = None,
    n_cells_expressed_threshold: int = 10,
) -> dict:
    """Build `adata_gt`/`adata_other` for `CSDEAnalysis`/`run_csde`.

    Reads a completed upstream CSDE annotation directory (produced by `scripts/export.py`
    + `scripts/annotate.py`).

    Label encoding in `.obs["prediction"]` / `.obs["annotation"]`: 1=target, 0=reference,
    2=other. Reads `config.json` (`cell_type_key`, `cell_type_of_interest`), `metadata.csv`
    (`sampling_weight` per annotated cell), and `annotations.json` (`is_correct` per cell) from
    `annotation_dir`.

    Parameters
    ----------
    annotation_dir
        Directory produced by upstream CSDE's `export.py`/`annotate.py`.
    sdata
        Already-loaded SpatialData object. If `None`, loaded from the path in `config.json`.
    layer
        AnnData layer to use for expression counts; `.X` is used if `None`.

    Returns
    -------
    dict with keys `adata_gt` (annotated cells; obs: `prediction`, `annotation`, `is_correct`,
    `sampling_weight`) and `adata_other` (all unannotated cells; obs: `prediction`). Both share
    the same gene set, filtered to genes expressed in at least `n_cells_expressed_threshold`
    predicted-target/reference annotated cells.
    """
    annotation_dir = Path(annotation_dir)

    with open(annotation_dir / "config.json") as f:
        config = json.load(f)
    cell_type_key = config["cell_type_key"]
    cell_type_of_interest = config["cell_type_of_interest"]

    ann_path = annotation_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"No annotations found at {ann_path}. Run upstream CSDE's scripts/annotate.py first."
        )
    with open(ann_path) as f:
        annotations = json.load(f)

    metadata = pd.read_csv(annotation_dir / "metadata.csv")
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    sampling_weights = metadata.set_index("cell_id")["sampling_weight"]

    if sdata is None:
        import spatialdata as sd

        sdata_path = config.get("sdata")
        if not sdata_path:
            raise ValueError(
                "No sdata path found in config.json. "
                "Pass sdata directly: prepare_csde_inputs(..., sdata=your_sdata_object)."
            )
        if not Path(sdata_path).exists():
            raise FileNotFoundError(
                f"SpatialData zarr not found at '{sdata_path}' (path stored in config.json). "
                "Either restore the zarr to that path, or pass sdata directly: "
                "prepare_csde_inputs(..., sdata=your_sdata_object)."
            )
        sdata = sd.read_zarr(sdata_path)

    adata = sdata["table"].copy()
    adata.obs_names = adata.obs_names.astype(str)
    adata = adata[adata.obs[cell_type_key].notna()].copy()

    is_coi = (adata.obs[cell_type_key] == cell_type_of_interest).values
    spatial_group = adata.obs[spatial_group_key].values

    prediction = np.full(len(adata), 2, dtype=int)
    prediction[is_coi & (spatial_group == spatial_group_target)] = 1
    prediction[is_coi & (spatial_group == spatial_group_reference)] = 0
    adata.obs["prediction"] = prediction

    annotated_ids = set(annotations.keys())
    annotated_mask = adata.obs_names.isin(annotated_ids)
    adata_gt = adata[annotated_mask].copy()

    is_correct_arr = np.array([annotations[cid] for cid in adata_gt.obs_names], dtype=bool)
    adata_gt.obs["is_correct"] = is_correct_arr

    is_coi_gt = (adata_gt.obs[cell_type_key] == cell_type_of_interest).values
    spatial_group_gt = adata_gt.obs[spatial_group_key].values
    annotation = np.full(len(adata_gt), 2, dtype=int)
    annotation[is_coi_gt & is_correct_arr & (spatial_group_gt == spatial_group_target)] = 1
    annotation[is_coi_gt & is_correct_arr & (spatial_group_gt == spatial_group_reference)] = 0
    adata_gt.obs["annotation"] = annotation

    adata_gt.obs["sampling_weight"] = adata_gt.obs_names.map(sampling_weights).values

    pred_mask = adata_gt.obs["annotation"].isin([0, 1])
    _sub = adata_gt[pred_mask]
    x = _sub.layers[layer] if layer is not None else _sub.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = x.astype(float)
    n_expressing = np.array((x >= 1).sum(0)).flatten()
    gene_mask = n_expressing >= n_cells_expressed_threshold

    adata_gt = adata_gt[:, gene_mask].copy()
    adata_other = adata[~annotated_mask][:, gene_mask].copy()

    return {"adata_gt": adata_gt, "adata_other": adata_other}
