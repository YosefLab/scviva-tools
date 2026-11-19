from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scviva.plotting.vision import (
    plot_selection_histplot,
    plot_signature_for_selection,
    plot_vision_autocorrelation,
    plot_vision_de_results,
)


@pytest.fixture
def adata_with_vision_results():
    n_obs, n_sigs = 20, 2
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n_obs, 5)).astype(float)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(5)])
    sig_cols = [f"sig{i}" for i in range(n_sigs)]
    obsm = {
        "spatial": rng.random((n_obs, 2)) * 100,
        "vision_signatures": pd.DataFrame(
            rng.normal(size=(n_obs, n_sigs)), columns=sig_cols, index=obs.index
        ),
    }
    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)

    adata.uns["vision_signature_scores"] = pd.DataFrame(
        {"c_prime": [0.8, -0.3], "fdr": [0.01, 0.2]}, index=sig_cols
    )
    adata.uns["vision_obs_df_scores"] = pd.DataFrame(
        {"c_prime": [0.5], "fdr": [0.04]}, index=["some_obs_col"]
    )

    var_name = "group"
    adata.uns[f"one_vs_all_signatures_{var_name}_scores"] = pd.DataFrame(
        rng.normal(size=(n_sigs, 2)), index=sig_cols, columns=["g0", "g1"]
    )
    adata.uns[f"one_vs_all_signatures_{var_name}_padj"] = pd.DataFrame(
        [[0.01, 0.6], [0.7, 0.02]], index=sig_cols, columns=["g0", "g1"]
    )
    return adata


def test_plot_signature_for_selection_returns_scatter_artifacts(adata_with_vision_results):
    p, ax, points = plot_signature_for_selection(
        adata_with_vision_results,
        signature="sig0",
        coords_obsm_key="spatial",
        s=10,
        vmin=None,
        vmax=None,
        figsize=(3, 3),
        cmap="viridis",
        colorbar=False,
    )
    assert points.shape == (adata_with_vision_results.n_obs, 2)
    assert ax is not None
    plt.close("all")


def test_plot_selection_histplot_runs_without_error(adata_with_vision_results, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    group = np.zeros(adata_with_vision_results.n_obs, dtype=int)
    group[:10] = 1
    plot_selection_histplot(adata_with_vision_results, signature="sig0", group=group)
    plt.close("all")


@pytest.mark.parametrize("plot_type", ["signatures", "observations"])
def test_plot_vision_autocorrelation_runs_without_error(
    adata_with_vision_results, monkeypatch, plot_type
):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_vision_autocorrelation(adata_with_vision_results, type=plot_type)
    plt.close("all")


def test_plot_vision_autocorrelation_rejects_invalid_type(adata_with_vision_results):
    with pytest.raises(ValueError, match="observations.*signatures"):
        plot_vision_autocorrelation(adata_with_vision_results, type="bogus")


def test_plot_vision_de_results_runs_without_error(adata_with_vision_results, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_vision_de_results(adata_with_vision_results, type="signatures", var="group")
    plt.close("all")


def test_plot_vision_de_results_requires_var(adata_with_vision_results):
    with pytest.raises(ValueError, match="var"):
        plot_vision_de_results(adata_with_vision_results, type="signatures", var=None)
