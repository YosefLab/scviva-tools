from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scviva.plotting._utils import prettify_axis


def plot_signature_for_selection(
    adata, signature, coords_obsm_key, s, vmin, vmax, figsize, cmap, colorbar
):
    """Plot signature scores on spatial coordinates for selection.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing:

        - ``obsm['vision_signatures']``: DataFrame of per-cell signature scores
        - ``obsm[coords_obsm_key]``: spatial coordinates (array or DataFrame)
    signature : str
        Name of the signature (column of ``obsm['vision_signatures']``) used to color cells.
    coords_obsm_key : str
        Key in ``adata.obsm`` with the spatial coordinates. The first two columns are used
        as x and y.
    s : float
        Marker size passed to :func:`matplotlib.pyplot.scatter`.
    vmin, vmax : float or None
        Limits of the color scale passed to :func:`matplotlib.pyplot.scatter`.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for the signature scores.
    colorbar : bool
        Whether to add a colorbar.

    Returns
    -------
    p : matplotlib.collections.PathCollection
        The scatter plot collection.
    ax : matplotlib.axes.Axes
        The axes the scores were plotted on.
    points : numpy.ndarray
        Array of shape ``(n_obs, 2)`` with the x and y coordinates of the plotted cells.
    """
    scores = adata.obsm["vision_signatures"]

    if isinstance(adata.obsm[coords_obsm_key], pd.DataFrame):
        coords = adata.obsm[coords_obsm_key].values
    else:
        coords = adata.obsm[coords_obsm_key]

    points = np.column_stack([coords[:, 0], coords[:, 1]])

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    prettify_axis(ax, spatial=True)
    p = plt.scatter(
        coords[:, 0], coords[:, 1], c=scores[signature], cmap=cmap, s=s, vmin=vmin, vmax=vmax
    )
    plt.title(signature)
    if colorbar:
        plt.colorbar()

    return p, ax, points


def plot_selection_histplot(adata, signature, group):
    """Plot histogram of signature scores for selection vs remainder.

    Note that this function modifies ``adata.obs``: it writes a ``'selected'`` column and,
    if not already present, a column named ``signature`` with the signature scores.

    Parameters
    ----------
    adata : AnnData
        AnnData object. Signature scores are read from ``obs[signature]`` if present,
        otherwise from ``obsm['vision_signatures'][signature]``.
    signature : str
        Name of the signature whose score distribution is plotted.
    group : array-like
        Per-cell indicator of length ``n_obs``; cells where ``group == 1`` are labeled
        ``"Selection"`` and all others ``"Remainder"``.
    """
    adata.obs["selected"] = pd.Series(
        np.where(np.asarray(group) == 1, "Selection", "Remainder"), index=adata.obs_names
    )

    if signature not in adata.obs:
        adata.obs[signature] = adata.obsm["vision_signatures"][signature]

    sns.histplot(
        data=adata.obs,
        x=signature,
        hue=adata.obs["selected"].tolist(),
        bins=30,
        palette={"Selection": "#FF7F00", "Remainder": "#1F78B4"},
    )
    plt.show()

    return


def plot_vision_autocorrelation(
    adata,
    type: Literal["observations"] | Literal["signatures"] | None = None,
    center: float | None = 0.5,
    figsize: tuple | None = (1, 10),
    cmap: str | None = "coolwarm",
    cbar: bool | None = True,
):
    """Plot vision autocorrelation results.

    Shows the consistency score (``c_prime``) of each observation column or signature as
    a heatmap, sorted in decreasing order; entries with FDR >= 0.05 are masked.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing ``uns['vision_obs_df_scores']`` (for
        ``type="observations"``) or ``uns['vision_signature_scores']`` (for
        ``type="signatures"``), each a DataFrame with ``c_prime`` and ``fdr`` columns.
    type : {"observations", "signatures"}
        Which autocorrelation results to plot. Must be set; ``None`` raises a
        ``ValueError``.
    center : float, default 0.5
        Value at which to center the colormap.
    figsize : tuple, default (1, 10)
        Figure size.
    cmap : str, default "coolwarm"
        Colormap for the heatmap.
    cbar : bool, default True
        Whether to draw a colorbar.
    """
    if type not in ["observations", "signatures"]:
        raise ValueError('The "type" variable should be one of ["observations", "signatures"].')

    type_str = "vision_obs_df_scores" if type == "observations" else "vision_signature_scores"

    masked_data = adata.uns[type_str][["c_prime"]].where(
        (adata.uns[type_str][["fdr"]] < 0.05).values
    )
    masked_data = masked_data.sort_values("c_prime", ascending=False)
    masked_data.columns = ["Consistency"]

    plt.figure(figsize=figsize)
    sns.heatmap(masked_data, annot=masked_data, cmap=cmap, fmt=".2f", cbar=cbar, center=center)
    plt.show()

    return


def plot_vision_de_results(
    adata,
    type: Literal["observations"] | Literal["signatures"] | None = None,
    var: str = None,
    center: float | None = 0.5,
    figsize: tuple | None = (3, 10),
    cmap: str | None = "coolwarm",
    cbar: bool | None = True,
):
    """Plot vision differential expression results.

    Shows one-vs-all scores as a heatmap; entries with p-value >= 0.05 are masked and
    significant entries are marked with ``*``.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the one-vs-all results for ``var``:

        - for ``type="observations"``: ``uns['one_vs_all_obs_cols_{var}_scores']`` and
          ``uns['one_vs_all_obs_cols_{var}_pvals']``
        - for ``type="signatures"``: ``uns['one_vs_all_signatures_{var}_scores']`` and
          ``uns['one_vs_all_signatures_{var}_padj']``
    type : {"observations", "signatures"}
        Which differential results to plot. Must be set; ``None`` raises a
        ``ValueError``.
    var : str
        Name of the categorical variable the one-vs-all comparison was run on. Must be
        set; ``None`` raises a ``ValueError``.
    center : float, default 0.5
        Value at which to center the colormap.
    figsize : tuple, default (3, 10)
        Figure size.
    cmap : str, default "coolwarm"
        Colormap for the heatmap.
    cbar : bool, default True
        Whether to draw a colorbar.
    """
    if var is None:
        raise ValueError('The "var" variable should be a categorical variable to plot.')

    if type not in ["observations", "signatures"]:
        raise ValueError('The "type" variable should be one of ["observations", "signatures"].')

    type_score_str = (
        f"one_vs_all_obs_cols_{var}_scores"
        if type == "observations"
        else f"one_vs_all_signatures_{var}_scores"
    )
    type_pval_str = (
        f"one_vs_all_obs_cols_{var}_pvals"
        if type == "observations"
        else f"one_vs_all_signatures_{var}_padj"
    )

    mask = adata.uns[type_pval_str] < 0.05

    plt.figure(figsize=figsize)
    sns.heatmap(
        adata.uns[type_score_str],
        mask=~mask,
        cmap=cmap,
        annot=mask.map(lambda x: "*" if x else ""),
        fmt="",
        cbar=cbar,
        center=center,
    )
    plt.show()

    return
