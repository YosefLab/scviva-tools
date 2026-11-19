"""Small shared plotting helpers used across plotting submodules."""

from __future__ import annotations

import matplotlib.pyplot as plt


def prettify_axis(ax, spatial: bool = False) -> None:
    """Hide the top/right spines and, for spatial plots, the tick labels."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if spatial:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Spatial1")
        plt.ylabel("Spatial2")
