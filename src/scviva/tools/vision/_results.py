from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scviva.tools.vision._constants import (
    DENDROGRAM_UNS_KEY,
    GENE_IMPORTANCE_UNS_KEY,
    META_DIFFERENTIAL_UNS_KEY,
    OBS_DF_SCORES_UNS_KEY,
    PROTEIN_AUTOCORR_UNS_KEY,
    PROTEIN_DIFFERENTIAL_UNS_KEY,
    SIG_CLUSTERS_UNS_KEY,
    SIGNATURE_DIFFERENTIAL_UNS_KEY,
    SIGNATURE_SCORES_UNS_KEY,
    SIGNATURES_OBSM_KEY,
)

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


@dataclass
class VisionResults:
    """Typed view over the ``adata.uns``/``adata.obsm`` keys written by analysis steps.

    All fields are direct references to the stored objects — no copy.
    Usually obtained via :attr:`~scviva.tools.vision.VisionAnalysis.results`
    (which calls :meth:`from_adata_uns`); a field is ``None`` when the
    analysis step that writes it has not been run (or produced no result).

    Parameters
    ----------
    signature_scores
        Per-cell signature scores (cells x signatures), from
        ``adata.obsm["vision_signatures"]``; written by
        :meth:`~scviva.tools.vision.VisionAnalysis.compute_signatures`.
    obs_df_scores
        Autocorrelation (Geary's C) table for ``adata.obs`` metadata columns,
        from ``adata.uns["vision_obs_df_scores"]``; written during
        :meth:`~scviva.tools.vision.VisionAnalysis.compute_differential_expression`.
    signature_autocorrelation
        Per-signature autocorrelation (Geary's C) table, from
        ``adata.uns["vision_signature_scores"]``; written by
        :meth:`~scviva.tools.vision.VisionAnalysis.compute_signatures`.
    signature_differential
        One-vs-all signature differential results, from
        ``adata.uns["vision_signature_differential"]``: a dict mapping each
        categorical ``adata.obs`` column to a :class:`~pandas.DataFrame` of
        per-group results.
    meta_differential
        One-vs-all differential results for ``adata.obs`` metadata columns,
        from ``adata.uns["vision_meta_differential"]``: a dict keyed by
        categorical ``adata.obs`` column.
    gene_importance
        Per-signature gene importance, from
        ``adata.uns["vision_gene_importance"]``: a dict mapping signature name
        to a :class:`~pandas.DataFrame` with ``importance`` and ``sign``
        columns indexed by gene.
    signature_clusters
        Signature cluster assignments, from ``adata.uns["vision_sig_clusters"]``:
        a dict mapping signature name to its cluster label.
    signature_dendrogram
        Hierarchical clustering of signatures as a Newick string, from
        ``adata.uns["vision_dendrogram"]``.
    protein_autocorrelation
        Per-protein autocorrelation (Geary's C) table, from
        ``adata.uns["vision_protein_autocorr"]``; only present when protein
        data is available.
    protein_differential
        One-vs-all protein differential results, from
        ``adata.uns["vision_protein_differential"]``: a dict mapping each
        categorical ``adata.obs`` column to a :class:`~pandas.DataFrame`;
        only present when protein data is available.
    """

    signature_scores: pd.DataFrame | None
    obs_df_scores: pd.DataFrame | None
    signature_autocorrelation: pd.DataFrame | None
    signature_differential: dict | None
    meta_differential: dict | None
    gene_importance: dict | None
    signature_clusters: dict | None
    signature_dendrogram: str | None
    protein_autocorrelation: Any | None
    protein_differential: dict | None

    @classmethod
    def from_adata_uns(cls, uns: dict, obsm: dict) -> VisionResults:
        """Construct from ``adata.uns`` and ``adata.obsm`` (the full top-level mappings).

        Parameters
        ----------
        uns
            The full ``adata.uns`` mapping.
        obsm
            The full ``adata.obsm`` mapping.
        """
        return cls(
            signature_scores=obsm.get(SIGNATURES_OBSM_KEY),
            obs_df_scores=uns.get(OBS_DF_SCORES_UNS_KEY),
            signature_autocorrelation=uns.get(SIGNATURE_SCORES_UNS_KEY),
            signature_differential=uns.get(SIGNATURE_DIFFERENTIAL_UNS_KEY),
            meta_differential=uns.get(META_DIFFERENTIAL_UNS_KEY),
            gene_importance=uns.get(GENE_IMPORTANCE_UNS_KEY),
            signature_clusters=uns.get(SIG_CLUSTERS_UNS_KEY),
            signature_dendrogram=uns.get(DENDROGRAM_UNS_KEY),
            protein_autocorrelation=uns.get(PROTEIN_AUTOCORR_UNS_KEY),
            protein_differential=uns.get(PROTEIN_DIFFERENTIAL_UNS_KEY),
        )
