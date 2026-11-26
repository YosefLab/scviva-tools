from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def match_reference_genes(
    st_var_names: Sequence[str],
    ref_var_names: Sequence[str],
    *,
    model_name: str,
    allow_subset: bool,
) -> np.ndarray | None:
    """Match the genes of a spatial dataset to those of a pretrained reference model.

    The gene-indexed parameters of a reference model (for example
    :class:`~scvi.model.CondSCVI` or :class:`~scviva.external.RNAStereoscope`) are reused by the
    spatial model, so the spatial data must be indexed by the same genes.

    Parameters
    ----------
    st_var_names
        ``var_names`` of the spatial AnnData.
    ref_var_names
        ``var_names`` the reference model was trained on.
    model_name
        Name of the spatial model, used in error messages.
    allow_subset
        Whether the spatial data may hold a subset of the reference genes, in any order. If
        ``False``, the genes must be identical and in the same order.

    Returns
    -------
    ``None`` if the genes are identical and in the same order. Otherwise, if ``allow_subset``,
    the position of every spatial gene in the reference.

    Raises
    ------
    ValueError
        If the spatial data has genes that are absent from the reference, or if
        ``allow_subset`` is ``False`` and the genes are not identical and in the same order.
    """
    st_index = pd.Index(st_var_names)
    ref_index = pd.Index(ref_var_names)
    if st_index.equals(ref_index):
        return None

    n_shared = int(st_index.isin(ref_index).sum())
    hint = (
        " No gene name is shared: check that both use the same identifiers "
        "(for example gene symbols or Ensembl IDs)."
        if n_shared == 0
        else ""
    )
    counts = (
        f"The spatial data has {len(st_index)} genes and the reference model "
        f"{len(ref_index)}, of which {n_shared} are shared."
    )

    if not allow_subset:
        if n_shared == len(ref_index) == len(st_index):
            raise ValueError(
                f"{model_name}: the genes of the spatial data are those of the reference model "
                "but in a different order. Reorder the spatial data before `setup_anndata`, "
                "for example `st_adata = st_adata[:, sc_model.adata.var_names].copy()`."
            )
        raise ValueError(
            f"{model_name}: the spatial data must have the same genes, in the same order, as "
            f"the reference model. {counts}{hint} The decoder of the reference normalizes "
            "expression over all of its genes, so it cannot be restricted to a subset: subset "
            "both datasets to their shared genes and train the reference model on them."
        )

    if not st_index.is_unique:
        raise ValueError(f"{model_name}: the gene names of the spatial data are not unique.")
    positions = ref_index.get_indexer(st_index)
    if (positions < 0).any():
        raise ValueError(
            f"{model_name}: {int((positions < 0).sum())} genes of the spatial data are absent "
            f"from the reference model. {counts}{hint} Subset the spatial data to the shared "
            "genes before `setup_anndata`, for example "
            "`st_adata = st_adata[:, st_adata.var_names.isin(sc_model.adata.var_names)].copy()`."
        )
    return positions
