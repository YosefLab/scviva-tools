from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._annotation import prepare_csde_inputs
from .tools._api import _build_and_fit, _format_results

if TYPE_CHECKING:
    import pandas as pd


class CSDEAnalysis:
    """Corrected Spatial Differential Expression between two cell populations.

    Uses prediction-powered inference (PPI) combining a large automated-annotation set
    with a small manually-validated subset.

    See Also
    --------
    scviva.tools.csde.tl.run_csde : underlying functional API.
    """

    def __init__(
        self,
        adata_pred,
        adata_gt,
        pred_cell_pop_key: str = "prediction",
        cell_pop_a=0,
        cell_pop_b=1,
        gt_key: str = "is_correct",
        layer_name: str | None = None,
        importance_weights: np.ndarray | None = None,
    ):
        self.adata_pred = adata_pred
        self.adata_gt = adata_gt
        self.pred_cell_pop_key = pred_cell_pop_key
        self.cell_pop_a = cell_pop_a
        self.cell_pop_b = cell_pop_b
        self.gt_key = gt_key
        self.layer_name = layer_name
        self.importance_weights = importance_weights
        self._model = None
        self._feature_names = None

    @classmethod
    def from_spatialdata(
        cls,
        sdata,
        annotation_dir,
        spatial_group_key: str = "spatial_group",
        spatial_group_target: int = 1,
        spatial_group_reference: int = 0,
        layer: str | None = None,
        n_cells_expressed_threshold: int = 10,
    ) -> CSDEAnalysis:
        """Build a `CSDEAnalysis` from a SpatialData object and annotation directory.

        The annotation directory is produced by upstream CSDE's `scripts/export.py` +
        `scripts/annotate.py` from https://github.com/YosefLab/CSDE — not reimplemented
        in scviva-tools; see the CSDE user guide page for what is and isn't ported.
        """
        inputs = prepare_csde_inputs(
            annotation_dir,
            sdata=sdata,
            spatial_group_key=spatial_group_key,
            spatial_group_target=spatial_group_target,
            spatial_group_reference=spatial_group_reference,
            layer=layer,
            n_cells_expressed_threshold=n_cells_expressed_threshold,
        )
        adata_gt, adata_other = inputs["adata_gt"], inputs["adata_other"]
        importance_weights = adata_gt.obs["sampling_weight"].to_numpy()
        return cls(
            adata_pred=adata_other,
            adata_gt=adata_gt,
            pred_cell_pop_key="prediction",
            cell_pop_a=0,
            cell_pop_b=1,
            gt_key="is_correct",
            layer_name=layer,
            importance_weights=importance_weights,
        )

    def fit(
        self, noise_model: str = "poisson", optimizer: str = "gd", **model_kwargs
    ) -> CSDEAnalysis:
        """Fit the PPI model. `noise_model`: `"poisson"`/`"nb"`. `optimizer`: `"gd"`/`"lbfgs"`."""
        self._model = _build_and_fit(
            self.adata_pred,
            self.adata_gt,
            self.pred_cell_pop_key,
            self.cell_pop_a,
            self.cell_pop_b,
            self.gt_key,
            layer_name=self.layer_name,
            importance_weights=self.importance_weights,
            noise_model=noise_model,
            optimizer=optimizer,
            **model_kwargs,
        )
        self._feature_names = list(self.adata_gt.var_names)
        return self

    def test_differential_expression(self, cond_thresh: float = np.inf) -> pd.DataFrame:
        """Test DE between `cell_pop_a`/`cell_pop_b`; `cond_thresh` caps ill-conditioned p-vals."""
        if self._model is None:
            raise RuntimeError("Call `.fit()` before `.test_differential_expression()`.")
        res = self._model.test_differential_expression(
            idx_a=1, feature_names=self._feature_names, cond_thresh=cond_thresh
        )
        return _format_results(res)
