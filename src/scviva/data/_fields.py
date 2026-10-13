from __future__ import annotations

import numpy as np
from scvi.data.fields import ObsmField


class SpatialCoordsField(ObsmField):
    """AnnData obsm field for 2D/3D spatial coordinates.

    Wraps scvi's ObsmField with coordinate dimensionality validation.
    Stores coordinates as-is (no normalization) — callers may normalize
    before registration if needed.

    Parameters
    ----------
    obsm_key
        Key to access the spatial coordinates in ``adata.obsm``.
    """

    def __init__(self, obsm_key: str) -> None:
        super().__init__(registry_key=obsm_key, attr_key=obsm_key)

    def get_field_data(self, adata):
        """Get the spatial coordinates from ``adata.obsm`` and validate their dimensionality.

        Parameters
        ----------
        adata
            AnnData object containing the coordinates in ``adata.obsm[obsm_key]``.

        Returns
        -------
        The coordinate array, with shape ``(n_obs, 2)`` or ``(n_obs, 3)``.

        Raises
        ------
        ValueError
            If the coordinates do not have 2 or 3 columns.
        """
        data = super().get_field_data(adata)
        if data.shape[1] not in (2, 3):
            raise ValueError(
                f"Spatial coordinates must be 2D or 3D, got shape {data.shape}. "
                f"Expected obsm['{self.attr_key}'] with 2 or 3 columns."
            )
        return data


class NeighborhoodGraphField(ObsmField):
    """AnnData obsm field for neighbor index/distance arrays.

    Stores dense numpy arrays. Squidpy outputs sparse CSR matrices —
    this field converts them to dense on registration (documented behavior).
    Keys: ``index_neighbor`` and ``distance_neighbor`` (matching upstream ResolVI).

    Parameters
    ----------
    obsm_key
        Key to access the neighbor graph in ``adata.obsm``.
    """

    def __init__(self, obsm_key: str) -> None:
        super().__init__(registry_key=obsm_key, attr_key=obsm_key)

    def get_field_data(self, adata):
        """Get the neighbor graph array from ``adata.obsm`` as a dense numpy array.

        Parameters
        ----------
        adata
            AnnData object containing the neighbor array in ``adata.obsm[obsm_key]``.

        Returns
        -------
        Dense :class:`~numpy.ndarray`; sparse inputs are converted with ``toarray()``.
        """
        data = super().get_field_data(adata)
        # Squidpy returns CSR; convert to dense for AnnDataManager compatibility
        if hasattr(data, "toarray"):
            data = data.toarray()
        return np.asarray(data)
