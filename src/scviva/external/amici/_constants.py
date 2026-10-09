from __future__ import annotations

from typing import NamedTuple


class _AMICIRegistryKeys(NamedTuple):
    COORD_KEY: str = "amici_coordinates"
    NN_IDX_KEY: str = "_amici_nn_idx"
    NN_DIST_KEY: str = "_amici_nn_dist"
    NN_X_KEY: str = "_amici_nn_x"


AMICI_REGISTRY_KEYS = _AMICIRegistryKeys()
