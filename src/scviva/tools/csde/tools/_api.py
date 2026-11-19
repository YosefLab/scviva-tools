import anndata
import numpy as np
import pandas as pd

from scviva.tools.csde._model_nb import NBIntercept
from scviva.tools.csde._model_poisson import PoissonIntercept


def _map_cell_types(obs: pd.DataFrame, cell_type_col: str, cell_pop_a, cell_pop_b) -> np.ndarray:
    """Map cell types to a 3-class representation: 0=cell_pop_a, 1=cell_pop_b, 2=other."""
    labels = np.full(len(obs), 2, dtype=int)
    if cell_pop_a not in obs[cell_type_col].values:
        raise ValueError(f"Cell population '{cell_pop_a}' not found in column '{cell_type_col}'")
    if cell_pop_b not in obs[cell_type_col].values:
        raise ValueError(f"Cell population '{cell_pop_b}' not found in column '{cell_type_col}'")
    labels[obs[cell_type_col] == cell_pop_a] = 0
    labels[obs[cell_type_col] == cell_pop_b] = 1
    return labels


def _get_X(adata: anndata.AnnData, layer_name: str | None) -> np.ndarray:
    X = adata.layers[layer_name] if layer_name else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(float)


def _align_genes(adata_gt: anndata.AnnData, adata_pred: anndata.AnnData) -> anndata.AnnData:
    """Reorder `adata_pred` to `adata_gt`'s gene order, requiring an identical gene set.

    `adata_gt.var_names` is the gene order used downstream for `feature_names`, so
    `adata_pred` must be aligned to it before its matrix is combined with `adata_gt`'s.
    """
    if set(adata_gt.var_names) != set(adata_pred.var_names):
        missing_in_pred = sorted(set(adata_gt.var_names) - set(adata_pred.var_names))
        missing_in_gt = sorted(set(adata_pred.var_names) - set(adata_gt.var_names))
        raise ValueError(
            "adata_gt and adata_pred must contain the same genes. "
            f"In adata_gt but not adata_pred: {missing_in_pred[:5]}"
            f"{'...' if len(missing_in_pred) > 5 else ''}. "
            f"In adata_pred but not adata_gt: {missing_in_gt[:5]}"
            f"{'...' if len(missing_in_gt) > 5 else ''}."
        )
    if not adata_gt.var_names.equals(adata_pred.var_names):
        adata_pred = adata_pred[:, adata_gt.var_names]
    return adata_pred


def _build_and_fit(
    adata_pred: anndata.AnnData,
    adata_gt: anndata.AnnData,
    pred_cell_pop_key: str,
    cell_pop_a,
    cell_pop_b,
    gt_key: str,
    layer_name: str | None = None,
    importance_weights: np.ndarray | None = None,
    noise_model: str = "poisson",
    **model_kwargs,
):
    """Build 3-class inputs, fit the PPI model, and compute its asymptotic distribution.

    Shared by the functional `run_csde()` (this module) and `CSDEAnalysis.fit()`
    (`scviva.tools.csde._analysis`), so both stay numerically identical.
    """
    y_pred_unl = _map_cell_types(adata_pred.obs, pred_cell_pop_key, cell_pop_a, cell_pop_b)
    y_pred_gt_set = _map_cell_types(adata_gt.obs, pred_cell_pop_key, cell_pop_a, cell_pop_b)

    y_gt = np.full(len(adata_gt), 2, dtype=int)
    is_correct = adata_gt.obs[gt_key].values.astype(bool)
    is_pred_a = (adata_gt.obs[pred_cell_pop_key] == cell_pop_a).values
    is_pred_b = (adata_gt.obs[pred_cell_pop_key] == cell_pop_b).values
    y_gt[is_pred_a & is_correct] = 0
    y_gt[is_pred_b & is_correct] = 1

    adata_pred = _align_genes(adata_gt, adata_pred)
    X_gt = _get_X(adata_gt, layer_name)
    X_unl = _get_X(adata_pred, layer_name)
    inputs_gt = (X_gt, y_gt)
    inputs_hat = (X_gt, y_pred_gt_set)
    inputs_unl = (X_unl, y_pred_unl)

    model_cls = {"poisson": PoissonIntercept, "nb": NBIntercept}.get(noise_model)
    if model_cls is None:
        raise ValueError(f"Unknown noise model: {noise_model}")

    model = model_cls(
        inputs_gt=inputs_gt,
        inputs_hat=inputs_hat,
        inputs_unl=inputs_unl,
        importance_weights=importance_weights,
        **model_kwargs,
    )
    model.fit(lambd_=None)
    model.get_asymptotic_distribution()
    return model


def _format_results(res: pd.DataFrame) -> pd.DataFrame:
    res = res.rename(columns={"beta": "log_fold_change", "pval": "p_value", "padj": "p_value_adj"})
    if "feature_name" in res.columns:
        res = res.set_index("feature_name")
    return res[["log_fold_change", "p_value", "p_value_adj"]]


def run_csde(
    adata_pred: anndata.AnnData,
    adata_gt: anndata.AnnData,
    pred_cell_pop_key: str,
    cell_pop_a,
    cell_pop_b,
    gt_key: str,
    layer_name: str | None = None,
    importance_weights: np.ndarray | None = None,
    noise_model: str = "poisson",
    **model_kwargs,
) -> pd.DataFrame:
    """Corrected spatial differential expression between two cell populations via PPI.

    See `scviva.tools.csde.CSDEAnalysis` for the class-based, SpatialData-integrated
    entry point built on top of this function.

    Parameters
    ----------
    adata_pred
        Cells with prediction-based (automated) population assignments only.
    adata_gt
        Cells with a manually-validated ground-truth subset.
    pred_cell_pop_key
        `obs` column with the prediction-based population labels (present in both AnnDatas).
    cell_pop_a, cell_pop_b
        Reference / target population labels (values of `pred_cell_pop_key`).
    gt_key
        Boolean `obs` column in `adata_gt` indicating whether the automated prediction was correct.
    layer_name
        `.layers` key holding raw counts; `.X` is used if `None`.
    importance_weights
        Optional 1-D array of sampling weights for `adata_gt`'s observations, normalized to
        sum to `n_obs` internally.
    noise_model
        `"poisson"` or `"nb"`.

    Returns
    -------
    DataFrame indexed by gene name with columns `log_fold_change`, `p_value`, `p_value_adj`.
    """
    model = _build_and_fit(
        adata_pred,
        adata_gt,
        pred_cell_pop_key,
        cell_pop_a,
        cell_pop_b,
        gt_key,
        layer_name=layer_name,
        importance_weights=importance_weights,
        noise_model=noise_model,
        **model_kwargs,
    )
    res = model.test_differential_expression(idx_a=1, feature_names=list(adata_gt.var_names))
    return _format_results(res)
