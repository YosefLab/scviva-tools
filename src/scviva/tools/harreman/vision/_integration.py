"""Cross-integration of VISION signature scoring with Hotspot gene modules.

Relates per-cell VISION signature scores (:mod:`scviva.tools.vision`) to
Hotspot gene-module assignments and activity scores
(:mod:`scviva.tools.harreman.hotspot`) via hypergeometric gene-overlap
enrichment and score correlation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


def _compute_sig_mod_enrichment(
    adata: AnnData, norm_data_key: str, signature_varm_key: str, use_super_modules: bool
):
    """Hypergeometric enrichment of signatures against Hotspot modules."""
    gene_modules_key = "gene_modules_sm" if use_super_modules else "gene_modules"

    use_raw = norm_data_key == "use_raw"
    genes = adata.raw.var.index if use_raw else adata.var_names

    sig_matrix = (
        adata.varm[signature_varm_key] if not use_raw else adata.raw.varm[signature_varm_key]
    )
    gene_modules = adata.uns[gene_modules_key]

    signatures = {}
    for signature in sig_matrix.columns:
        if all(x in sig_matrix[signature].unique().tolist() for x in [-1, 1]):
            signatures[signature + "_UP"] = sig_matrix[sig_matrix[signature] == 1].index.tolist()
            signatures[signature + "_DOWN"] = sig_matrix[
                sig_matrix[signature] == -1
            ].index.tolist()
        else:
            signatures[signature] = sig_matrix[sig_matrix[signature] != 0].index.tolist()

    pvals_df = pd.DataFrame(
        np.nan, index=list(signatures.keys()), columns=list(gene_modules.keys())
    )
    stats_df = pd.DataFrame(
        np.nan, index=list(signatures.keys()), columns=list(gene_modules.keys())
    )

    sig_mod_df = pd.DataFrame(index=genes)

    universe = adata.var_names[adata.var["local_autocorrelation"]].tolist()
    signatures = {
        sig: [g for g in sig_genes if g in universe] for sig, sig_genes in signatures.items()
    }

    for signature, sig_genes in signatures.items():
        for module, mod_genes in gene_modules.items():
            sig_mod_genes = list(set(sig_genes) & set(mod_genes))
            M = len(universe)
            n = len(sig_genes)
            N = len(mod_genes)
            x = len(sig_mod_genes)
            pval = hypergeom.sf(x - 1, M, n, N)
            if pval < 0.05:
                sig_mod_name = signature + "_OVERLAP_" + module
                sig_mod_df[sig_mod_name] = 0
                sig_mod_df.loc[sig_mod_genes, sig_mod_name] = 1.0
            e_overlap = n * N / M if M != 0 else 0
            stat = np.log2(x / e_overlap) if e_overlap != 0 else 0
            pvals_df.loc[signature, module] = pval
            stats_df.loc[signature, module] = stat

    stacked_pvals = pvals_df.stack()
    fdr_values = multipletests(stacked_pvals.values, method="fdr_bh")[1]
    fdr_df = pd.Series(fdr_values, index=stacked_pvals.index).unstack()

    if use_raw:
        adata.raw.varm["signatures_overlap"] = sig_mod_df
    else:
        adata.varm["signatures_overlap"] = sig_mod_df

    return pvals_df, stats_df, fdr_df


def _compute_sig_mod_correlation(adata: AnnData, method: str, use_super_modules: bool):
    """Pearson or Spearman correlation between signature and module scores."""
    module_scores_key = "super_module_scores" if use_super_modules else "module_scores"

    signatures = adata.obsm["vision_signatures"].columns.tolist()
    modules = adata.obsm[module_scores_key].columns.tolist()

    cor_pval_df = pd.DataFrame(index=modules)
    cor_coef_df = pd.DataFrame(index=modules)

    for signature in signatures:
        correlation_values = []
        pvals = []
        for module in modules:
            sig_scores = adata.obsm["vision_signatures"][signature]
            mod_scores = adata.obsm[module_scores_key][module]
            if method == "pearson":
                corr, pval = pearsonr(sig_scores, mod_scores)
            else:
                corr, pval = spearmanr(sig_scores, mod_scores)
            correlation_values.append(corr)
            pvals.append(pval)
        cor_coef_df[signature] = correlation_values
        cor_pval_df[signature] = pvals

    stacked_cor_pvals = cor_pval_df.stack()
    fdr_values = multipletests(stacked_cor_pvals.values, method="fdr_bh")[1]
    cor_fdr_df = pd.Series(fdr_values, index=stacked_cor_pvals.index).unstack()

    return cor_coef_df, cor_pval_df, cor_fdr_df


def integrate_vision_hotspot_results(
    adata: AnnData,
    cor_method: Literal["pearson", "spearman"] = "pearson",
    use_super_modules: bool = False,
) -> None:
    """
    Integrate VISION signature scoring with Hotspot module assignments.

    Requires that VISION signature scoring has already been run (producing
    ``adata.obsm['vision_signatures']``, ``adata.uns['norm_data_key']``, and
    ``adata.uns['signature_varm_key']`` — e.g. via
    :meth:`~scviva.tools.vision.VisionAnalysis.compute_signatures`) and that
    Hotspot modules have been created via
    :func:`~scviva.tools.harreman.hotspot.create_modules` and scored via
    :func:`~scviva.tools.harreman.hotspot.calculate_module_scores` (or
    :func:`~scviva.tools.harreman.hotspot.calculate_super_module_scores` when
    ``use_super_modules=True``).

    For each (signature, module) pair the function computes:

    * A hypergeometric enrichment test of signature gene overlap with module
      genes (log2 fold-enrichment, raw p-value, BH-FDR).
    * Pearson or Spearman correlation between per-cell signature scores and
      per-cell module activity scores.

    An overlap signature matrix (genes present in significant signature-module
    intersections) is also computed and stored for downstream use.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Must contain:

        - ``obsm['vision_signatures']``: per-cell signature scores produced by
          :func:`scviva.tools.vision.tools.signature.compute_signatures_anndata`.
        - ``uns['norm_data_key']``: expression layer key used for signature
          scoring (``None``, ``"use_raw"``, or a layer name).
        - ``uns['signature_varm_key']``: key in ``adata.varm`` for the
          gene x signature scoring matrix.
        - ``uns['gene_modules']`` (or ``uns['gene_modules_sm']`` when
          ``use_super_modules=True``): module -> gene-list mapping produced by
          :func:`calculate_module_scores` / :func:`calculate_super_module_scores`.
        - ``obsm['module_scores']`` (or ``obsm['super_module_scores']``): per-cell
          module activity scores.
    cor_method : {'pearson', 'spearman'}, default 'pearson'
        Correlation method used to relate signature scores to module scores.
    use_super_modules : bool, default False
        If ``True``, use super-module gene lists and scores (``gene_modules_sm``
        / ``super_module_scores``) instead of the standard module equivalents.

    Returns
    -------
    None
        Results are stored in-place on ``adata``:

        - ``uns['sig_mod_enrichment_stats']``: log2 fold-enrichment
          (signatures x modules).
        - ``uns['sig_mod_enrichment_pvals']``: raw hypergeometric p-values
          (signatures x modules).
        - ``uns['sig_mod_enrichment_FDR']``: BH-corrected FDR values
          (signatures x modules).
        - ``uns['sig_mod_correlation_coefs']``: correlation coefficients
          (modules x signatures).
        - ``uns['sig_mod_correlation_pvals']``: raw correlation p-values
          (modules x signatures).
        - ``uns['sig_mod_correlation_FDR']``: BH-corrected FDR values
          (modules x signatures).
        - ``uns['cor_method']``: the correlation method used.
        - ``varm['signatures_overlap']``: binary gene x overlap-signature matrix
          for signature-module intersections with p < 0.05.
        - ``obsm['signature_modules_overlap']``: per-cell scores for each
          significant signature-module overlap.
    """
    from scviva.tools.vision.tools.signature import compute_signatures_anndata

    gene_modules_key = "gene_modules_sm" if use_super_modules else "gene_modules"

    if "vision_signatures" not in adata.obsm:
        raise ValueError(
            "adata.obsm['vision_signatures'] not found. "
            "Run VISION signature scoring before calling this function."
        )
    if gene_modules_key not in adata.uns or len(adata.uns[gene_modules_key]) == 0:
        raise ValueError(
            f"adata.uns['{gene_modules_key}'] not found or empty. "
            "Run create_modules() (and calculate_super_module_scores() if "
            "use_super_modules=True) before calling this function."
        )
    if cor_method not in ("pearson", "spearman"):
        raise ValueError(f"Invalid cor_method '{cor_method}'. Choose 'pearson' or 'spearman'.")

    norm_data_key = adata.uns["norm_data_key"]
    signature_varm_key = adata.uns["signature_varm_key"]

    start = time.time()
    logger.info("Integrating VISION and Hotspot results...")

    pvals_df, stats_df, fdr_df = _compute_sig_mod_enrichment(
        adata, norm_data_key, signature_varm_key, use_super_modules
    )
    adata.uns["sig_mod_enrichment_stats"] = stats_df
    adata.uns["sig_mod_enrichment_pvals"] = pvals_df
    adata.uns["sig_mod_enrichment_FDR"] = fdr_df

    adata.uns["cor_method"] = cor_method
    cor_coef_df, cor_pval_df, cor_fdr_df = _compute_sig_mod_correlation(
        adata, cor_method, use_super_modules
    )
    adata.uns["sig_mod_correlation_coefs"] = cor_coef_df
    adata.uns["sig_mod_correlation_pvals"] = cor_pval_df
    adata.uns["sig_mod_correlation_FDR"] = cor_fdr_df

    # Score the per-module overlap genes as signatures.  We temporarily call
    # compute_signatures_anndata with a different varm key, then move the result
    # to the correct obsm slot and restore the original vision_signatures scores.
    _saved_vision_sigs = adata.obsm.get("vision_signatures")
    compute_signatures_anndata(
        adata,
        norm_data_key,
        "signatures_overlap",
    )
    adata.obsm["signature_modules_overlap"] = adata.obsm.pop("vision_signatures")
    if _saved_vision_sigs is not None:
        adata.obsm["vision_signatures"] = _saved_vision_sigs
    # Restore uns keys that compute_signatures_anndata overwrote
    adata.uns["norm_data_key"] = norm_data_key
    adata.uns["signature_varm_key"] = signature_varm_key

    logger.info(
        "Finished integrating VISION and Hotspot results in %.3f seconds", time.time() - start
    )
