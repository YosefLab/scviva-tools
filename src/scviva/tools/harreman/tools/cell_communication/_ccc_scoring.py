import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.stats import norm, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from scviva.tools.harreman.preprocessing.anndata import (
    counts_from_anndata_for_genes,
)
from scviva.tools.harreman.tools.knn import make_weights_non_redundant
from scviva.utils import resolve_device, stats_dtype

from ._metabolite_scoring import compute_metabolite_cs
from ._stats import compute_max_cs, flatten, z_to_pval_fdr


def compute_cell_communication(
    adata: AnnData,
    layer_key_p_test: Literal["use_raw"] | str | None = None,
    layer_key_np_test: Literal["use_raw"] | str | None = None,
    model: str = None,
    center_counts_for_np_test: bool | None = False,
    subset_gene_pairs: str | None = None,
    M: int | None = 1000,
    seed: int | None = 42,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None = "both",
    mean: Literal["algebraic"] | Literal["geometric"] | None = "algebraic",
    check_analytic_null: bool | None = False,
    device: torch.device | str = "auto",
    verbose: bool | None = False,
) -> None:
    """Compute spatially informed cell-type-agnostic CCC scores.

    Scores and significance are computed across all gene pairs using both parametric and
    non-parametric statistical tests.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Required fields include:
            - `uns["gene_pairs"]`: list of gene pairs to evaluate.
            - `uns["gene_pairs_per_metabolite"]`: dictionary mapping metabolites to gene pairs.
            - `obsp["weights"]`: sparse matrix encoding spatial cell-cell proximity.
            - (Optional) `uns["LR_database"]`: interaction metadata for pathway annotation.
            - (Optional) `uns["sample_key"]`: if modeling includes sample-specific factors.
    layer_key_p_test : str or "use_raw", optional
        Data layer to use for the parametric test. If `"use_raw"`, uses `adata.raw`.
    layer_key_np_test : str or "use_raw", optional
        Data layer to use for the non-parametric test. If `"use_raw"`, uses `adata.raw`.
    model : str, optional
        Normalization model to use for centering gene expression. Options include "none",
        "normal", "bernoulli", or "danb".
    center_counts_for_np_test : bool, optional (default: False)
        Whether to center expression counts using the specified model before non-parametric
        testing.
    subset_gene_pairs : list, optional
        If provided, restricts the analysis to this subset of gene pairs.
    M : int, optional (default: 1000)
        Number of permutations to use if `permutation_test` is True.
    seed : int, optional (default: 42)
        Random seed for permutation reproducibility.
    test : {'parametric', 'non-parametric', 'both'}, optional (default: 'both')
        Specifies which statistical test(s) to run.
    mean : {'algebraic', 'geometric'}, optional (default: 'algebraic')
        Averaging method for multi-gene interactions.
    check_analytic_null : bool, optional (default: False)
        Whether to evaluate Z-scores under an analytic null distribution using permutation
        Z-scores.
    device : torch.device, optional
        PyTorch device to run computations on. Defaults to CUDA if available.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
        Results are stored in the following `adata.uns` fields:
            - `uns["ccc_results"]["p"]`: Parametric test results (scores, Z, p-values, FDR).
            - `uns["ccc_results"]["np"]`: Non-parametric results (scores, p-values, FDR).
            - `uns["lc_zs"]`: Symmetric matrix of ligand-receptor Z-scores across genes.
            - `uns["gene_pair_dict"]`: Dictionary mapping metabolites to gene pair indices.
            - `uns["D"]`: Vector of total node degrees per cell (spatial connectivity).
            - `uns["genes"]`: Ordered list of involved genes.
            - `uns["gene_pairs_ind"]`: Index-referenced version of `uns["gene_pairs"]`.
    """
    start = time.time()
    device = resolve_device(device)
    if verbose:
        print("Starting cell-cell communication analysis...")

    adata.uns["ccc_results"] = {}

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError(
            'The "test" variable should be one of ["both", "parametric", "non-parametric"].'
        )

    if mean not in ["algebraic", "geometric"]:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')

    adata.uns["layer_key_p_test"] = layer_key_p_test
    adata.uns["layer_key_np_test"] = layer_key_np_test
    adata.uns["model"] = model
    adata.uns["center_counts_for_np_test"] = center_counts_for_np_test
    adata.uns["mean"] = mean

    run_cell_communication_analysis(
        adata,
        layer_key_p_test,
        layer_key_np_test,
        model,
        center_counts_for_np_test,
        subset_gene_pairs,
        M,
        seed,
        test,
        mean,
        check_analytic_null,
        device,
        verbose,
    )

    if verbose:
        print("Obtaining communication results...")
    get_cell_communication_results(
        adata,
        adata.uns["genes"],
        layer_key_p_test,
        layer_key_np_test,
        model,
        adata.uns["D"],
        test,
        device,
    )

    if verbose:
        print(
            "Finished computing cell-cell communication analysis in %.3f seconds"
            % (time.time() - start)
        )

    return


def run_cell_communication_analysis(
    adata: AnnData,
    layer_key_p_test: Literal["use_raw"] | str | None,
    layer_key_np_test: Literal["use_raw"] | str | None,
    model: str | None,
    center_counts_for_np_test: bool | None,
    subset_gene_pairs: str | None,
    M: int | None,
    seed: int | None,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None,
    mean: Literal["algebraic"] | Literal["geometric"] | None,
    check_analytic_null: bool | None,
    device: torch.device | str,
    verbose: bool | None,
) -> None:
    """Run the cell-type-agnostic CCC score and significance workflow."""
    # `obs_names` never differs between `adata` and `adata.raw`, and `Raw` has
    # no `.obs`/`.obs_names.values` to read regardless.
    cells = adata.obs_names.values.astype(str)

    sample_specific = "sample_key" in adata.uns

    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    genes = list(np.unique(list(flatten(adata.uns["gene_pairs"]))))
    adata.uns["genes"] = genes
    adata.uns["cells"] = cells

    # Map gene_pairs to index
    gene_pairs_ind = []
    for pair in gene_pairs:
        idx1 = (
            [genes.index(g) for g in pair[0]]
            if isinstance(pair[0], list)
            else genes.index(pair[0])
        )
        idx2 = (
            [genes.index(g) for g in pair[1]]
            if isinstance(pair[1], list)
            else genes.index(pair[1])
        )
        gene_pairs_ind.append((idx1, idx2))
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

    # Compute weights
    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=stats_dtype(device), device=device),
        torch.Size(weights.shape),
        device=device,
    )

    # Compute node degree
    row_degrees = torch.sparse.sum(weights, dim=1).to_dense()
    col_degrees = torch.sparse.sum(weights, dim=0).to_dense()
    D = row_degrees + col_degrees

    adata.uns["D"] = D.cpu().numpy()

    gene_pairs_per_metabolite = adata.uns["gene_pairs_per_metabolite"]

    metabolite_gene_pair_df = pd.DataFrame.from_dict(
        gene_pairs_per_metabolite, orient="index"
    ).reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})

    metabolite_gene_pair_df["gene_pair"] = metabolite_gene_pair_df["gene_pair"].apply(
        lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr]
    )
    metabolite_gene_pair_df["gene_type"] = metabolite_gene_pair_df["gene_type"].apply(
        lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr]
    )

    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df["metabolite"],
            metabolite_gene_pair_df.explode("gene_pair")["gene_pair"],
            metabolite_gene_pair_df.explode("gene_type")["gene_type"],
        ],
        axis=1,
    )
    metabolite_gene_pair_df = metabolite_gene_pair_df.reset_index(drop=True)

    if "LR_database" in adata.uns.keys():
        LR_database = adata.uns["LR_database"]
        df_merged = pd.merge(
            metabolite_gene_pair_df,
            LR_database,
            left_on="metabolite",
            right_on="interaction_name",
            how="left",
        )
        LR_df = df_merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df["metabolite"][
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)
        ] = LR_df["pathway_name"]

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = (
            group["gene_pair"]
            .apply(lambda gp: gene_pairs.index(gp) if gp in gene_pairs else None)
            .dropna()
            .tolist()
        )
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs

    adata.uns["gene_pair_dict"] = gene_pair_dict

    if test in ["parametric", "both"]:
        if verbose:
            print("Running the parametric test...")

        adata.uns["ccc_results"]["p"] = {"gp": {}, "m": {}}

        Wtot2 = torch.tensor((weights.data**2).sum(), dtype=stats_dtype(device), device=device)

        # Load counts
        counts = counts_from_anndata_for_genes(
            adata, genes, layer_key_p_test, dense=True, cells=cells
        )
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)
        num_umi = counts.sum(dim=0)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        # Standardize counts
        from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

        counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

        counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        # Compute CCC scores
        WX2t = torch.sparse.mm(weights, counts_2.T)
        WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        cs_gp = (counts_1.T * WX2t).sum(0) + (counts_1.T * WtX2t).sum(0)
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs], device=device)
        cs_gp[same_gene_mask] = cs_gp[same_gene_mask] / 2
        adata.uns["ccc_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        # Compute metabolite-level scores
        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
        adata.uns["ccc_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        # Compute second moments
        WX1t = torch.sparse.mm(weights, counts_1.T)
        WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
        eg2_a = (WX1t + WtX1t).pow(2).sum(dim=0)
        eg2_b = (WX2t + WtX2t).pow(2).sum(dim=0)
        eg2s_gp = (eg2_a, eg2_b)

        # Z-score computation
        Z_gp, Z_m = compute_p_results(cs_gp, cs_m, gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict)
        # Convert tensors to numpy for statsmodels and pandas
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        # Compute p-values and FDRs
        Z_pvals_gp, FDR_gp = z_to_pval_fdr(Z_gp_np)
        Z_pvals_m, FDR_m = z_to_pval_fdr(Z_m_np)

        # Store in AnnData
        adata.uns["ccc_results"]["p"]["gp"]["Z"] = Z_gp_np
        adata.uns["ccc_results"]["p"]["gp"]["Z_pval"] = Z_pvals_gp
        adata.uns["ccc_results"]["p"]["gp"]["Z_FDR"] = FDR_gp
        adata.uns["ccc_results"]["p"]["m"]["Z"] = Z_m_np
        adata.uns["ccc_results"]["p"]["m"]["Z_pval"] = Z_pvals_m
        adata.uns["ccc_results"]["p"]["m"]["Z_FDR"] = FDR_m

        # Symmetric LC Z-score matrix
        genes_ = [
            tuple(i) if isinstance(i, list) else i
            for i in pd.Series([g for pair in gene_pairs for g in pair]).drop_duplicates()
        ]
        gene_pairs_ = [
            (tuple(a) if isinstance(a, list) else a, tuple(b) if isinstance(b, list) else b)
            for a, b in gene_pairs
        ]
        lc_zs = pd.DataFrame(np.zeros((len(genes_), len(genes_))), index=genes_, columns=genes_)
        for i, (g1, g2) in enumerate(gene_pairs_):
            lc_zs.iloc[genes_.index(g1), genes_.index(g2)] = Z_gp_np[i]
        # Force diagonal to 0 and symmetrize
        lc_zs_arr = lc_zs.to_numpy(copy=True)
        np.fill_diagonal(lc_zs_arr, 0)
        lc_zs.loc[:, :] = lc_zs_arr
        adata.uns["lc_zs"] = (lc_zs + lc_zs.T) / 2

        if verbose:
            print("Parametric test finished.")

    if test in ["non-parametric", "both"]:
        if verbose:
            print("Running the non-parametric test...")

        adata.uns["ccc_results"]["np"] = {"gp": {}, "m": {}}

        # Load counts
        counts = counts_from_anndata_for_genes(
            adata, genes, layer_key_np_test, dense=True, cells=cells
        )
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

            counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
            from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        n_cells = counts_1.shape[1]
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs], device=device)

        if center_counts_for_np_test and test == "both":
            adata.uns["ccc_results"]["np"]["gp"]["cs"] = np.array(
                adata.uns["ccc_results"]["p"]["gp"]["cs"]
            )
            adata.uns["ccc_results"]["np"]["m"]["cs"] = np.array(
                adata.uns["ccc_results"]["p"]["m"]["cs"]
            )
        else:
            WX2t = torch.sparse.mm(weights, counts_2.T)
            WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
            cs_gp = (counts_1.T * WX2t).sum(0) + (counts_1.T * WtX2t).sum(0)
            cs_gp[same_gene_mask] = cs_gp[same_gene_mask] / 2
            adata.uns["ccc_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
            adata.uns["ccc_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        perm_cs_gp_a = torch.zeros(
            (counts_1.shape[0], M), dtype=stats_dtype(device), device=device
        )
        perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)
        perm_cs_m_a = torch.zeros(
            (len(gene_pair_dict), M), dtype=stats_dtype(device), device=device
        )
        perm_cs_m_b = torch.zeros_like(perm_cs_m_a)

        if check_analytic_null:
            gp_zs_perm_array = torch.zeros_like(perm_cs_gp_a)
            gp_pvals_perm_array = torch.zeros_like(perm_cs_gp_a)
            m_zs_perm_array = torch.zeros_like(perm_cs_m_a)
            m_pvals_perm_array = torch.zeros_like(perm_cs_m_a)

        torch.manual_seed(seed)
        for i in tqdm(range(M), desc="Permutation test"):
            idx = torch.randperm(n_cells, device=device)

            c1_perm_a = counts_1.clone()
            c2_perm_a = counts_2[:, idx]
            c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]

            WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
            WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
            cs_a = (c1_perm_a.T * WX2t_a).sum(0) + (c1_perm_a.T * WtX2t_a).sum(0)
            cs_a[same_gene_mask] = cs_a[same_gene_mask] / 2
            perm_cs_gp_a[:, i] = cs_a

            cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=False)
            perm_cs_m_a[:, i] = cs_m_a

            c2_perm_b = counts_2.clone()
            c1_perm_b = counts_1[:, idx]
            c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]

            WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
            WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
            cs_b = (c1_perm_b.T * WX2t_b).sum(0) + (c1_perm_b.T * WtX2t_b).sum(0)
            cs_b[same_gene_mask] = cs_b[same_gene_mask] / 2
            perm_cs_gp_b[:, i] = cs_b

            cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=False)
            perm_cs_m_b[:, i] = cs_m_b

            if check_analytic_null:
                Z_gp_perm, Z_m_perm = compute_p_results(
                    (cs_a, cs_b), (cs_m_a, cs_m_b), gene_pairs_ind, Wtot2, eg2s_gp, gene_pair_dict
                )
                gp_zs_perm_array[:, i] = Z_gp_perm
                gp_pvals_perm_array[:, i] = torch.tensor(
                    norm.sf(Z_gp_perm.cpu().numpy()), dtype=stats_dtype(device), device=device
                )
                m_zs_perm_array[:, i] = Z_m_perm
                m_pvals_perm_array[:, i] = torch.tensor(
                    norm.sf(Z_m_perm.cpu().numpy()), dtype=stats_dtype(device), device=device
                )

        adata.uns["ccc_results"]["np"]["gp"]["perm_cs_a"] = perm_cs_gp_a.detach().cpu().numpy()
        adata.uns["ccc_results"]["np"]["gp"]["perm_cs_b"] = perm_cs_gp_b.detach().cpu().numpy()
        adata.uns["ccc_results"]["np"]["m"]["perm_cs_a"] = perm_cs_m_a.detach().cpu().numpy()
        adata.uns["ccc_results"]["np"]["m"]["perm_cs_b"] = perm_cs_m_b.detach().cpu().numpy()

        x_gp_a = (perm_cs_gp_a > cs_gp[:, None]).sum(dim=1)
        x_gp_b = (perm_cs_gp_b > cs_gp[:, None]).sum(dim=1)
        x_m_a = (perm_cs_m_a > cs_m[:, None]).sum(dim=1)
        x_m_b = (perm_cs_m_b > cs_m[:, None]).sum(dim=1)

        pvals_gp_a = (x_gp_a + 1).float() / (M + 1)
        pvals_gp_b = (x_gp_b + 1).float() / (M + 1)
        pvals_m_a = (x_m_a + 1).float() / (M + 1)
        pvals_m_b = (x_m_b + 1).float() / (M + 1)

        pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b)
        pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b)

        adata.uns["ccc_results"]["np"]["gp"]["pval"] = pvals_gp.cpu().numpy()
        adata.uns["ccc_results"]["np"]["gp"]["FDR"] = multipletests(
            pvals_gp.cpu().numpy(), method="fdr_bh"
        )[1]
        adata.uns["ccc_results"]["np"]["m"]["pval"] = pvals_m.cpu().numpy()
        adata.uns["ccc_results"]["np"]["m"]["FDR"] = multipletests(
            pvals_m.cpu().numpy(), method="fdr_bh"
        )[1]

        if check_analytic_null:
            adata.uns["ccc_results"]["np"]["analytic_null"] = {
                "gp_zs_perm": gp_zs_perm_array.detach().cpu().numpy(),
                "gp_pvals_perm": gp_pvals_perm_array.detach().cpu().numpy(),
                "m_zs_perm": m_zs_perm_array.detach().cpu().numpy(),
                "m_pvals_perm": m_pvals_perm_array.detach().cpu().numpy(),
            }

    if verbose:
        print("Non-parametric test finished.")

    return


def select_significant_interactions(
    adata: AnnData,
    ct_aware: bool | None = False,
    test: Literal["parametric"] | Literal["non-parametric"] | None = "parametric",
    use_FDR: bool | None = True,
    threshold: float | None = 0.05,
) -> None:
    """Select significant gene pairs or metabolite-mediated interactions.

    Selection is based on FDR/p-value thresholds and optional cell-type-aware tests.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing:
        - ``uns['ccc_results']`` or ``uns['ct_ccc_results']``, each of which includes:
            * ``cell_com_df_gp``: DataFrame with statistics for gene pairs
            * ``cell_com_df_m``:  DataFrame with statistics for metabolites
    ct_aware : bool, default False
        If True, use cell-type–aware CCC results (``uns['ct_ccc_results']``).
        If False, use cell-type-agnostic CCC results (``uns['ccc_results']``).
    test : {"parametric", "non-parametric"}, default "parametric"
        Determines which statistical columns to use:
        - Parametric: ``Z_FDR`` / ``Z_pval``, ``C_p``
        - Non-parametric: ``FDR_np`` / ``pval_np``, ``C_np``
    use_FDR : bool, default True
        If True, threshold significance using FDR values.
        If False, use raw p-values.
    threshold : float, default 0.05
        Significance cutoff applied to the selected statistic (FDR or p-value).
    """
    ccc_key = "ct_ccc_results" if ct_aware else "ccc_results"
    sig_key = "FDR" if use_FDR else "pval"

    if test == "parametric":
        FDR_values_gp = adata.uns[ccc_key]["cell_com_df_gp"][f"Z_{sig_key}"].values
        C_values_gp = adata.uns[ccc_key]["cell_com_df_gp"]["C_p"].values
        FDR_values_m = adata.uns[ccc_key]["cell_com_df_m"][f"Z_{sig_key}"].values
        C_values_m = adata.uns[ccc_key]["cell_com_df_m"]["C_p"].values
    elif test == "non-parametric":
        FDR_values_gp = adata.uns[ccc_key]["cell_com_df_gp"][f"{sig_key}_np"].values
        C_values_gp = adata.uns[ccc_key]["cell_com_df_gp"]["C_np"].values
        FDR_values_m = adata.uns[ccc_key]["cell_com_df_m"][f"{sig_key}_np"].values
        C_values_m = adata.uns[ccc_key]["cell_com_df_m"]["C_np"].values
    else:
        raise ValueError('The "test" variable should be one of ["parametric", "non-parametric"].')

    # Gene pair
    adata.uns[ccc_key]["cell_com_df_gp"]["selected"] = (
        (FDR_values_gp < threshold) & (C_values_gp > 0)
        if test == "non-parametric"
        else (FDR_values_gp < threshold)
    )
    cell_com_df_gp = adata.uns[ccc_key]["cell_com_df_gp"]
    adata.uns[ccc_key]["cell_com_df_gp_sig"] = cell_com_df_gp[cell_com_df_gp["selected"]].copy()

    # Metabolite
    adata.uns[ccc_key]["cell_com_df_m"]["selected"] = (
        (FDR_values_m < threshold) & (C_values_m > 0)
        if test == "non-parametric"
        else (FDR_values_m < threshold)
    )
    cell_com_df_m = adata.uns[ccc_key]["cell_com_df_m"]
    adata.uns[ccc_key]["cell_com_df_m_sig"] = cell_com_df_m[cell_com_df_m["selected"]].copy()

    return


def compute_interacting_cell_scores(
    adata: str | AnnData,
    center_counts_for_np_test: bool | None = False,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None = "both",
    restrict_significance: Literal["gene pairs"]
    | Literal["metabolites"]
    | Literal["both"]
    | None = "both",
    compute_significance: Literal["parametric"]
    | Literal["non-parametric"]
    | Literal["both"]
    | None = "both",
    M: int | None = 1000,
    seed: int | None = 42,
    check_analytic_null: bool | None = False,
    device: torch.device | str = "auto",
    verbose: bool | None = False,
) -> None:
    """
    Compute interacting cell scores for gene pairs and metabolites.

    Parameters
    ----------
    adata : AnnData or str
        AnnData object containing:
        - ``uns['model']`` and ``uns['mean']`` (expression normalization model)
        - ``uns['gene_pairs']``, ``uns['gene_pairs_per_metabolite']``
        - ``obsp['weights']``: sparse spatial weight matrix
        - ``uns['ccc_results']`` (for significance filtering)
    center_counts_for_np_test : bool, optional
        If True, center/normalize counts prior to the non-parametric test.
    test : {"parametric", "non-parametric", "both"}
        Which interacting cell score tests to compute.
    restrict_significance : {"gene pairs", "metabolites", "both"}
        Use only significant gene pairs/metabolites from CCC results.
    compute_significance : {"parametric", "non-parametric", "both"}
        Whether to compute significance (p-values, FDR) in each test.
    M : int, default 1000
        Number of permutations for the non-parametric test.
    seed : int, default 42
        Random seed for permutation reproducibility.
    check_analytic_null : bool, default False
        If True, evaluate the analytic null distribution during permutations.
    device : torch.device
        CPU or GPU device for tensor operations.
    verbose : bool, default False
        Print status updates.

    Returns
    -------
    None
        Results are stored in ``adata.uns['interacting_cell_results']``.
    """
    start = time.time()
    device = resolve_device(device)
    if verbose:
        print("Computing gene pair and metabolite scores...")

    adata.uns["interacting_cell_results"] = {}

    model = adata.uns["model"]
    mean = adata.uns["mean"]

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError(
            'The "test" variable should be one of ["both", "parametric", "non-parametric"].'
        )

    if restrict_significance is not None and restrict_significance not in [
        "both",
        "gene pairs",
        "metabolites",
    ]:
        raise ValueError(
            'The "restrict_significance" variable should be one of '
            '["both", "gene pairs", "metabolites"].'
        )

    if compute_significance is not None and compute_significance not in [
        "both",
        "parametric",
        "non-parametric",
    ]:
        raise ValueError(
            'The "compute_significance" variable should be one of '
            '["both", "parametric", "non-parametric"].'
        )

    sample_specific = "sample_key" in adata.uns

    layer_key_p_test = adata.uns.get("layer_key_p_test", None)
    layer_key_np_test = adata.uns.get("layer_key_np_test", None)
    use_raw = (layer_key_p_test == "use_raw") and (layer_key_np_test == "use_raw")

    gene_pairs = adata.uns.get("gene_pairs", None)
    gene_pairs_per_metabolite = adata.uns["gene_pairs_per_metabolite"]

    def to_tuple(x):
        # Recursively convert lists to tuples
        if isinstance(x, list):
            return tuple(to_tuple(i) for i in x)
        return x

    metabolite_gene_pair_df = pd.DataFrame.from_dict(
        gene_pairs_per_metabolite, orient="index"
    ).reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})
    metabolite_gene_pair_df["gene_pair"] = metabolite_gene_pair_df["gene_pair"].apply(
        lambda arr: [(to_tuple(gp[0]), to_tuple(gp[1])) for gp in arr]
    )
    metabolite_gene_pair_df["gene_type"] = metabolite_gene_pair_df["gene_type"].apply(
        lambda arr: [(to_tuple(gt[0]), to_tuple(gt[1])) for gt in arr]
    )
    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df["metabolite"],
            metabolite_gene_pair_df.explode("gene_pair")["gene_pair"],
            metabolite_gene_pair_df.explode("gene_type")["gene_type"],
        ],
        axis=1,
    ).reset_index(drop=True)

    if "LR_database" in adata.uns:
        LR_database = adata.uns["LR_database"]
        df_merged = pd.merge(
            metabolite_gene_pair_df,
            LR_database,
            left_on="metabolite",
            right_on="interaction_name",
            how="left",
        )
        LR_df = df_merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df["metabolite"][
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite)
        ] = LR_df["pathway_name"]

    if restrict_significance in ["both", "gene pairs"]:
        cell_com_gp_df = adata.uns["ccc_results"]["cell_com_df_gp_sig"].copy()
        cell_com_gp_df[["Gene 1", "Gene 2"]] = cell_com_gp_df[["Gene 1", "Gene 2"]].map(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

        gene_pairs_set = {tuple(x) for x in cell_com_gp_df[["Gene 1", "Gene 2"]].values}
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df["gene_pair"].isin(gene_pairs_set)
        ]

    if restrict_significance in ["both", "metabolites"]:
        cell_com_m_df = adata.uns["ccc_results"]["cell_com_df_m_sig"].copy()
        metabolite_set = set(cell_com_m_df["Metabolite"].values)
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df["metabolite"].isin(metabolite_set)
        ]

    genes = adata.uns["genes"]
    gene_pairs_sig = []
    if gene_pairs:
        for g1, g2 in gene_pairs:
            g1 = tuple(g1) if isinstance(g1, list) else g1
            g2 = tuple(g2) if isinstance(g2, list) else g2
            if not metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"] == (g1, g2)].empty:
                gene_pairs_sig.append((g1, g2))

    adata.uns["gene_pairs_sig"] = gene_pairs_sig

    gene_pairs_sig_ind = []
    for g1, g2 in gene_pairs_sig:
        idx1 = tuple([genes.index(g) for g in g1]) if isinstance(g1, tuple) else genes.index(g1)
        idx2 = tuple([genes.index(g) for g in g2]) if isinstance(g2, tuple) else genes.index(g2)
        gene_pairs_sig_ind.append((idx1, idx2))

    adata.uns["gene_pairs_sig_ind"] = gene_pairs_sig_ind

    if "barcode_key" in adata.uns:
        barcode_key = adata.uns["barcode_key"]
        cells = pd.Series(adata.obs[barcode_key].tolist())
    else:
        cells = adata.obs_names if not use_raw else adata.raw.obs_names

    # Compute weights
    weights = make_weights_non_redundant(adata.obsp["weights"]).tocoo()
    weights = torch.sparse_coo_tensor(
        torch.tensor(np.vstack((weights.row, weights.col)), dtype=torch.long, device=device),
        torch.tensor(weights.data, dtype=stats_dtype(device), device=device),
        torch.Size(weights.shape),
        device=device,
    )

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = (
            group["gene_pair"]
            .apply(lambda gp: gene_pairs_sig.index(gp) if gp in gene_pairs_sig else None)
            .dropna()
            .tolist()
        )
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    metabolites = list(gene_pair_dict.keys())

    adata.uns["metabolites"] = metabolites

    gene_pairs_sig_names = [
        "_".join("_".join(g) if isinstance(g, tuple) else g for g in gp) for gp in gene_pairs_sig
    ]

    adata.uns["gene_pairs_sig_names"] = gene_pairs_sig_names

    if test in ["parametric", "both"]:
        if verbose:
            print("Running the parametric test...")

        adata.uns["interacting_cell_results"]["p"] = {"gp": {}, "m": {}}

        Wtot2 = torch.tensor((weights.data**2).sum(), dtype=stats_dtype(device), device=device)

        # Load counts
        counts = counts_from_anndata_for_genes(
            adata, genes, layer_key_p_test, dense=True, cells=cells
        )
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)
        num_umi = counts.sum(dim=0)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_sig_ind:
            if isinstance(idx1, tuple):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, tuple):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

        counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
        from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

        counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        # Compute CCC scores
        WX2t = torch.sparse.mm(weights, counts_2.T)
        WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
        cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)
        cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
        adata.uns["interacting_cell_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        # Compute metabolite-level scores
        cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
        adata.uns["interacting_cell_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        if compute_significance in ["parametric", "both"]:
            # Compute second moments
            WX1t = torch.sparse.mm(weights, counts_1.T)
            WtX1t = torch.sparse.mm(weights.transpose(0, 1), counts_1.T)
            eg2_a = (WX1t + WtX1t).pow(2)
            eg2_b = (WX2t + WtX2t).pow(2)
            eg2s_gp = (eg2_a, eg2_b)

            Z_gp, Z_m = compute_p_int_cell_results_no_ct(
                cs_gp, cs_m, gene_pairs_sig_ind, Wtot2, eg2s_gp, gene_pair_dict
            )

            Z_gp_np = Z_gp.detach().cpu().numpy()
            Z_m_np = Z_m.detach().cpu().numpy()
            # Compute p-values and FDRs
            Z_pvals_gp, FDR_gp = z_to_pval_fdr(Z_gp_np)
            Z_pvals_m, FDR_m = z_to_pval_fdr(Z_m_np)

            adata.uns["interacting_cell_results"]["p"]["gp"]["Z"] = Z_gp_np
            adata.uns["interacting_cell_results"]["p"]["gp"]["Z_pval"] = Z_pvals_gp
            adata.uns["interacting_cell_results"]["p"]["gp"]["Z_FDR"] = FDR_gp
            adata.uns["interacting_cell_results"]["p"]["m"]["Z"] = Z_m_np
            adata.uns["interacting_cell_results"]["p"]["m"]["Z_pval"] = Z_pvals_m
            adata.uns["interacting_cell_results"]["p"]["m"]["Z_FDR"] = FDR_m

            # P-value
            mask_gp = adata.uns["interacting_cell_results"]["p"]["gp"]["Z_pval"] < 0.05
            mask_m = adata.uns["interacting_cell_results"]["p"]["m"]["Z_pval"] < 0.05

            cs_gp_sig = adata.uns["interacting_cell_results"]["p"]["gp"]["cs"].copy()
            cs_m_sig = adata.uns["interacting_cell_results"]["p"]["m"]["cs"].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns["interacting_cell_results"]["p"]["gp"]["cs_sig_pval"] = cs_gp_sig
            adata.uns["interacting_cell_results"]["p"]["m"]["cs_sig_pval"] = cs_m_sig

            # FDR
            mask_gp = adata.uns["interacting_cell_results"]["p"]["gp"]["Z_FDR"] < 0.05
            mask_m = adata.uns["interacting_cell_results"]["p"]["m"]["Z_FDR"] < 0.05

            cs_gp_sig = adata.uns["interacting_cell_results"]["p"]["gp"]["cs"].copy()
            cs_m_sig = adata.uns["interacting_cell_results"]["p"]["m"]["cs"].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns["interacting_cell_results"]["p"]["gp"]["cs_sig_FDR"] = cs_gp_sig
            adata.uns["interacting_cell_results"]["p"]["m"]["cs_sig_FDR"] = cs_m_sig

        if verbose:
            print("Parametric test finished.")

    if test in ["non-parametric", "both"]:
        if verbose:
            print("Running the non-parametric test...")

        adata.uns["interacting_cell_results"]["np"] = {"gp": {}, "m": {}}

        # Load counts
        counts = counts_from_anndata_for_genes(
            adata, genes, layer_key_np_test, dense=True, cells=cells
        )
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_sig_ind:
            if isinstance(idx1, tuple):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, tuple):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

            counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
            from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        n_cells = counts_1.shape[1]
        same_gene_mask = torch.tensor([g1 == g2 for g1, g2 in gene_pairs_sig], device=device)

        if center_counts_for_np_test and test == "both":
            adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = np.array(
                adata.uns["interacting_cell_results"]["p"]["gp"]["cs"]
            )
            adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = np.array(
                adata.uns["interacting_cell_results"]["p"]["m"]["cs"]
            )
        else:
            WX2t = torch.sparse.mm(weights, counts_2.T)
            WtX2t = torch.sparse.mm(weights.transpose(0, 1), counts_2.T)
            cs_gp = (counts_1.T * WX2t) + (counts_1.T * WtX2t)
            cs_gp[:, same_gene_mask] = cs_gp[:, same_gene_mask] / 2
            adata.uns["interacting_cell_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
            adata.uns["interacting_cell_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        if compute_significance in ["non-parametric", "both"]:
            perm_cs_gp_a = torch.zeros(
                (n_cells, counts_1.shape[0], M), dtype=stats_dtype(device), device=device
            )
            perm_cs_gp_b = torch.zeros_like(perm_cs_gp_a)
            perm_cs_m_a = torch.zeros(
                (n_cells, len(gene_pair_dict), M), dtype=stats_dtype(device), device=device
            )
            perm_cs_m_b = torch.zeros_like(perm_cs_m_a)

            if check_analytic_null:
                gp_zs_perm_array = torch.zeros_like(perm_cs_gp_a)
                gp_pvals_perm_array = torch.zeros_like(perm_cs_gp_a)
                m_zs_perm_array = torch.zeros_like(perm_cs_m_a)
                m_pvals_perm_array = torch.zeros_like(perm_cs_m_a)

            torch.manual_seed(seed)
            for i in tqdm(range(M), desc="Permutation test"):
                idx = torch.randperm(n_cells, device=device)

                c1_perm_a = counts_1.clone()
                c2_perm_a = counts_2[:, idx]
                c1_perm_a[same_gene_mask] = counts_1[same_gene_mask, :][:, idx]

                WX2t_a = torch.sparse.mm(weights, c2_perm_a.T)
                WtX2t_a = torch.sparse.mm(weights.transpose(0, 1), c2_perm_a.T)
                cs_a = (c1_perm_a.T * WX2t_a) + (c1_perm_a.T * WtX2t_a)
                cs_a[:, same_gene_mask] = cs_a[:, same_gene_mask] / 2
                perm_cs_gp_a[:, :, i] = cs_a

                cs_m_a = compute_metabolite_cs(cs_a, gene_pair_dict, interacting_cell_scores=True)
                perm_cs_m_a[:, :, i] = cs_m_a

                c2_perm_b = counts_2.clone()
                c1_perm_b = counts_1[:, idx]
                c2_perm_b[same_gene_mask] = counts_2[same_gene_mask, :][:, idx]

                WX2t_b = torch.sparse.mm(weights, c2_perm_b.T)
                WtX2t_b = torch.sparse.mm(weights.transpose(0, 1), c2_perm_b.T)
                cs_b = (c1_perm_b.T * WX2t_b) + (c1_perm_b.T * WtX2t_b)
                cs_b[:, same_gene_mask] = cs_b[:, same_gene_mask] / 2
                perm_cs_gp_b[:, :, i] = cs_b

                cs_m_b = compute_metabolite_cs(cs_b, gene_pair_dict, interacting_cell_scores=True)
                perm_cs_m_b[:, :, i] = cs_m_b

                if check_analytic_null:
                    Z_gp_perm, Z_m_perm = compute_p_results(
                        (cs_a, cs_b),
                        (cs_m_a, cs_m_b),
                        gene_pairs_sig_ind,
                        Wtot2,
                        eg2s_gp,
                        gene_pair_dict,
                    )
                    gp_zs_perm_array[:, :, i] = Z_gp_perm
                    gp_pvals_perm_array[:, :, i] = torch.tensor(
                        norm.sf(Z_gp_perm.cpu().numpy()), dtype=stats_dtype(device), device=device
                    )
                    m_zs_perm_array[:, :, i] = Z_m_perm
                    m_pvals_perm_array[:, :, i] = torch.tensor(
                        norm.sf(Z_m_perm.cpu().numpy()), dtype=stats_dtype(device), device=device
                    )

            adata.uns["interacting_cell_results"]["np"]["gp"]["perm_cs_a"] = (
                perm_cs_gp_a.detach().cpu().numpy()
            )
            adata.uns["interacting_cell_results"]["np"]["gp"]["perm_cs_b"] = (
                perm_cs_gp_b.detach().cpu().numpy()
            )
            adata.uns["interacting_cell_results"]["np"]["m"]["perm_cs_a"] = (
                perm_cs_m_a.detach().cpu().numpy()
            )
            adata.uns["interacting_cell_results"]["np"]["m"]["perm_cs_b"] = (
                perm_cs_m_b.detach().cpu().numpy()
            )

            x_gp_a = (perm_cs_gp_a > cs_gp[:, :, None]).sum(dim=2)
            x_gp_b = (perm_cs_gp_b > cs_gp[:, :, None]).sum(dim=2)
            x_m_a = (perm_cs_m_a > cs_m[:, :, None]).sum(dim=2)
            x_m_b = (perm_cs_m_b > cs_m[:, :, None]).sum(dim=2)

            pvals_gp_a = (x_gp_a + 1).float() / (M + 1)
            pvals_gp_b = (x_gp_b + 1).float() / (M + 1)
            pvals_m_a = (x_m_a + 1).float() / (M + 1)
            pvals_m_b = (x_m_b + 1).float() / (M + 1)

            pvals_gp = torch.where(pvals_gp_a > pvals_gp_b, pvals_gp_a, pvals_gp_b)
            pvals_m = torch.where(pvals_m_a > pvals_m_b, pvals_m_a, pvals_m_b)

            pvals_gp = pvals_gp.cpu().numpy()
            pvals_m = pvals_m.cpu().numpy()

            adata.uns["interacting_cell_results"]["np"]["gp"]["pval"] = pvals_gp
            adata.uns["interacting_cell_results"]["np"]["gp"]["FDR"] = multipletests(
                pvals_gp.flatten(), method="fdr_bh"
            )[1].reshape(pvals_gp.shape)
            adata.uns["interacting_cell_results"]["np"]["m"]["pval"] = pvals_m
            adata.uns["interacting_cell_results"]["np"]["m"]["FDR"] = multipletests(
                pvals_m.flatten(), method="fdr_bh"
            )[1].reshape(pvals_m.shape)

            if check_analytic_null:
                adata.uns["interacting_cell_results"]["np"]["analytic_null"] = {
                    "gp_zs_perm": gp_zs_perm_array.detach().cpu().numpy(),
                    "gp_pvals_perm": gp_pvals_perm_array.detach().cpu().numpy(),
                    "m_zs_perm": m_zs_perm_array.detach().cpu().numpy(),
                    "m_pvals_perm": m_pvals_perm_array.detach().cpu().numpy(),
                }

            # P-value
            mask_gp = adata.uns["interacting_cell_results"]["np"]["gp"]["pval"] < 0.05
            mask_m = adata.uns["interacting_cell_results"]["np"]["m"]["pval"] < 0.05

            cs_gp_sig = adata.uns["interacting_cell_results"]["np"]["gp"]["cs"].copy()
            cs_m_sig = adata.uns["interacting_cell_results"]["np"]["m"]["cs"].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns["interacting_cell_results"]["np"]["gp"]["cs_sig_pval"] = cs_gp_sig
            adata.uns["interacting_cell_results"]["np"]["m"]["cs_sig_pval"] = cs_m_sig

            # FDR
            mask_gp = adata.uns["interacting_cell_results"]["np"]["gp"]["FDR"] < 0.05
            mask_m = adata.uns["interacting_cell_results"]["np"]["m"]["FDR"] < 0.05

            cs_gp_sig = adata.uns["interacting_cell_results"]["np"]["gp"]["cs"].copy()
            cs_m_sig = adata.uns["interacting_cell_results"]["np"]["m"]["cs"].copy()

            cs_gp_sig[~mask_gp] = np.nan
            cs_m_sig[~mask_m] = np.nan
            adata.uns["interacting_cell_results"]["np"]["gp"]["cs_sig_FDR"] = cs_gp_sig
            adata.uns["interacting_cell_results"]["np"]["m"]["cs_sig_FDR"] = cs_m_sig

        if verbose:
            print("Non-parametric test finished.")

    if verbose:
        print(
            "Finished computing gene pair and metabolite scores in %.3f seconds"
            % (time.time() - start)
        )

    return


def compute_p_results(
    C_gp: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    C_m: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    gene_pairs_ind: list[tuple[Any, Any]],
    Wtot2: torch.Tensor,
    eg2s_gp: tuple[torch.Tensor, torch.Tensor],
    gene_pair_dict: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute parametric Z-scores for gene pairs and metabolites."""
    device = Wtot2.device

    # Convert indices
    same_gene_mask = torch.tensor(
        [
            (isinstance(g1, int) and isinstance(g2, int) and g1 == g2)
            or (isinstance(g1, list) and isinstance(g2, list) and sorted(g1) == sorted(g2))
            for g1, g2 in gene_pairs_ind
        ],
        device=device,
    )

    # Unpack second moments
    EG2_a = eg2s_gp[0].clone()
    EG2_b = eg2s_gp[1].clone()
    EG2_a[same_gene_mask] = Wtot2
    EG2_b[same_gene_mask] = Wtot2

    stdG_a = torch.sqrt(EG2_a)
    stdG_b = torch.sqrt(EG2_b)
    stdG_a[stdG_a == 0] = 1
    stdG_b[stdG_b == 0] = 1

    # Compute gene-pair Z-scores
    if isinstance(C_gp, tuple):
        C_gp_0, C_gp_1 = C_gp
        z_0 = C_gp_0 / stdG_a
        z_1 = C_gp_1 / stdG_b
        mask = torch.abs(z_0) < torch.abs(z_1)
        Z_gp = torch.where(mask, z_0, z_1)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)
    else:
        C_gp = C_gp
        z_a = C_gp / stdG_a
        z_b = C_gp / stdG_b
        mask = torch.abs(z_a) < torch.abs(z_b)
        Z_gp = torch.where(mask, z_a, z_b)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)

    # Compute metabolite-level expected variance
    EG2_m = compute_metabolite_cs(EG2_gp, gene_pair_dict, interacting_cell_scores=False)
    if not isinstance(EG2_m, torch.Tensor):
        EG2_m = torch.tensor(EG2_m, device=device, dtype=stats_dtype(device))

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    # Compute metabolite Z-scores
    if isinstance(C_m, tuple):
        C_m_0, C_m_1 = C_m
        z_0 = C_m_0 / stdG_m
        z_1 = C_m_1 / stdG_m
        Z_m = torch.where(torch.abs(z_0) < torch.abs(z_1), z_0, z_1)
    else:
        Z_m = C_m / stdG_m

    return Z_gp, Z_m


def compute_p_int_cell_results_no_ct(
    C_gp: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    C_m: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    gene_pairs_ind: list[tuple[Any, Any]],
    Wtot2: torch.Tensor,
    eg2s_gp: tuple[torch.Tensor, torch.Tensor],
    gene_pair_dict: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interacting-cell parametric results without cell type stratification."""
    device = Wtot2.device

    # Convert indices
    same_gene_mask = torch.tensor(
        [
            (isinstance(g1, int) and isinstance(g2, int) and g1 == g2)
            or (isinstance(g1, list) and isinstance(g2, list) and sorted(g1) == sorted(g2))
            for g1, g2 in gene_pairs_ind
        ],
        device=device,
    )

    # Unpack second moments
    EG2_a = eg2s_gp[0].clone()
    EG2_b = eg2s_gp[1].clone()
    EG2_a[:, same_gene_mask] = Wtot2
    EG2_b[:, same_gene_mask] = Wtot2

    stdG_a = torch.sqrt(EG2_a)
    stdG_b = torch.sqrt(EG2_b)
    stdG_a[stdG_a == 0] = 1
    stdG_b[stdG_b == 0] = 1

    # Compute gene-pair Z-scores
    if isinstance(C_gp, tuple):
        C_gp_0, C_gp_1 = C_gp
        z_0 = C_gp_0 / stdG_a
        z_1 = C_gp_1 / stdG_b
        mask = torch.abs(z_0) < torch.abs(z_1)
        Z_gp = torch.where(mask, z_0, z_1)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)
    else:
        C_gp = C_gp
        z_a = C_gp / stdG_a
        z_b = C_gp / stdG_b
        mask = torch.abs(z_a) < torch.abs(z_b)
        Z_gp = torch.where(mask, z_a, z_b)
        EG2_gp = torch.where(mask, EG2_a, EG2_b)

    # Compute metabolite-level expected variance
    EG2_m = compute_metabolite_cs(EG2_gp, gene_pair_dict, interacting_cell_scores=True)
    if not isinstance(EG2_m, torch.Tensor):
        EG2_m = torch.tensor(EG2_m, device=device, dtype=stats_dtype(device))

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    # Compute metabolite Z-scores
    if isinstance(C_m, tuple):
        C_m_0, C_m_1 = C_m
        z_0 = C_m_0 / stdG_m
        z_1 = C_m_1 / stdG_m
        Z_m = torch.where(torch.abs(z_0) < torch.abs(z_1), z_0, z_1)
    else:
        Z_m = C_m / stdG_m

    return (Z_gp, Z_m)


def get_cell_communication_results(
    adata: AnnData,
    genes: list[str],
    layer_key_p_test: Literal["use_raw"] | str | None,
    layer_key_np_test: Literal["use_raw"] | str | None,
    model: str | None,
    D: np.ndarray | torch.Tensor,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None,
    device: torch.device | str,
) -> None:
    """Assemble cell-type-agnostic communication result dataframes."""
    gene_pairs = adata.uns["gene_pairs"]
    gene_pairs_ind = adata.uns["gene_pairs_ind"]
    gene_pair_dict = adata.uns["gene_pair_dict"]

    sample_specific = "sample_key" in adata.uns

    if isinstance(D, np.ndarray):
        D = torch.tensor(D, dtype=stats_dtype(device), device=device)

    # Initialize dataframes
    cell_com_df_gp = pd.DataFrame(gene_pairs, columns=["Gene 1", "Gene 2"])
    cell_com_df_m = pd.DataFrame({"Metabolite": list(gene_pair_dict.keys())})

    if test in ["parametric", "both"]:
        suffix = "p"
        # Gene pair
        c_values = adata.uns["ccc_results"][suffix]["gp"]["cs"]
        z_values = adata.uns["ccc_results"][suffix]["gp"]["Z"]
        p_values = adata.uns["ccc_results"][suffix]["gp"]["Z_pval"]
        fdr_values = adata.uns["ccc_results"][suffix]["gp"]["Z_FDR"]
        cell_com_df_gp[f"C_{suffix}"] = c_values
        cell_com_df_gp["Z"] = z_values
        cell_com_df_gp["Z_pval"] = p_values
        cell_com_df_gp["Z_FDR"] = fdr_values

        counts = counts_from_anndata_for_genes(adata, genes, layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)
        num_umi = counts.sum(dim=0)
        from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

        counts_std = standardize_counts(adata, counts, model, num_umi, sample_specific)

        c_values_norm = normalize_values(counts_std, gene_pairs_ind, c_values, D)
        adata.uns["ccc_results"][suffix]["gp"]["cs_norm"] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f"C_norm_{suffix}"] = c_values_norm.cpu().numpy()

        # Metabolite
        c_values = adata.uns["ccc_results"][suffix]["m"]["cs"]
        z_values = adata.uns["ccc_results"][suffix]["m"]["Z"]
        p_values = adata.uns["ccc_results"][suffix]["m"]["Z_pval"]
        fdr_values = adata.uns["ccc_results"][suffix]["m"]["Z_FDR"]
        cell_com_df_m[f"C_{suffix}"] = c_values
        cell_com_df_m["Z"] = z_values
        cell_com_df_m["Z_pval"] = p_values
        cell_com_df_m["Z_FDR"] = fdr_values

    if test in ["non-parametric", "both"]:
        suffix = "np"
        # Gene pair
        c_values = adata.uns["ccc_results"][suffix]["gp"]["cs"]
        p_values = adata.uns["ccc_results"][suffix]["gp"]["pval"]
        fdr_values = adata.uns["ccc_results"][suffix]["gp"]["FDR"]
        cell_com_df_gp[f"C_{suffix}"] = c_values
        cell_com_df_gp[f"pval_{suffix}"] = p_values
        cell_com_df_gp[f"FDR_{suffix}"] = fdr_values

        counts = counts_from_anndata_for_genes(adata, genes, layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)
        if adata.uns.get("center_counts_for_np_test", False):
            num_umi = counts.sum(dim=0)
            from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

            counts = standardize_counts(adata, counts, model, num_umi, sample_specific)

        c_values_norm = normalize_values(counts, gene_pairs_ind, c_values, D)
        adata.uns["ccc_results"][suffix]["gp"]["cs_norm"] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f"C_norm_{suffix}"] = c_values_norm.cpu().numpy()

        # Metabolite
        c_values = adata.uns["ccc_results"][suffix]["m"]["cs"]
        p_values = adata.uns["ccc_results"][suffix]["m"]["pval"]
        fdr_values = adata.uns["ccc_results"][suffix]["m"]["FDR"]
        cell_com_df_m[f"C_{suffix}"] = c_values
        cell_com_df_m[f"pval_{suffix}"] = p_values
        cell_com_df_m[f"FDR_{suffix}"] = fdr_values

    adata.uns["ccc_results"]["cell_com_df_gp"] = cell_com_df_gp
    adata.uns["ccc_results"]["cell_com_df_m"] = cell_com_df_m

    return


def normalize_values(
    counts: torch.Tensor,
    gene_pairs_ind: list[tuple[Any, Any]],
    lcs: torch.Tensor | np.ndarray,
    D: torch.Tensor,
) -> torch.Tensor:
    """Normalize communication scores (lcs) using maximum possible score estimates."""
    lc_maxs = compute_max_cs(D, counts, gene_pairs_ind)
    lc_maxs = torch.where(lc_maxs == 0, torch.tensor(1.0, device=lc_maxs.device), lc_maxs)
    if isinstance(lcs, np.ndarray):
        lcs = torch.tensor(lcs, dtype=lc_maxs.dtype, device=lc_maxs.device)
    c_values_norm = lcs / lc_maxs
    c_values_norm = torch.where(
        torch.isinf(c_values_norm), torch.tensor(1.0, device=c_values_norm.device), c_values_norm
    )
    return c_values_norm


def compute_interaction_module_correlation(
    adata: AnnData,
    cor_method: Literal["pearson"] | Literal["spearman"] | None = "pearson",
    test: Literal["parametric"] | Literal["non-parametric"] | None = None,
    interaction_type: Literal["metabolite"] | Literal["gene_pair"] | None = "metabolite",
    only_sig_values: bool | None = False,
    normalize_values: bool | None = False,
    use_FDR: bool | None = True,
    use_super_modules: bool | None = False,
    ct_aware: bool | None = None,
) -> None:
    """Compute correlations between interacting cell scores and module scores.

    Parameters
    ----------
    adata : AnnData
        Must contain:
        - ``uns['interacting_cell_results']`` with parametric or non-parametric scores
        - ``obsm['module_scores']`` or ``obsm['super_module_scores']``
        - ``uns['metabolites']`` or ``uns['gene_pairs_sig_names']``
    cor_method : {"pearson", "spearman"}, default "pearson"
        Statistical method used to compute correlations.
    test : {"parametric", "non-parametric"}
        Which interacting-cell score set to use.
        - `"parametric"` uses ``uns['interacting_cell_results']['p']``
        - `"non-parametric"` uses ``uns['interacting_cell_results']['np']``
    interaction_type : {"metabolite", "gene_pair"}, default "metabolite"
        Select whether to correlate:
        - metabolite scores, or
        - gene pair scores.
    only_sig_values : bool, default False
        If True, use only significant interacting cell score values (`cs_sig_pval` or
        `cs_sig_FDR`).
    normalize_values : bool, default False
        Apply min-max normalization to interacting cell score values per interaction.
    use_FDR : bool, default True
        If ``only_sig_values=True``, determines whether to filter by FDR or raw p-values.
    use_super_modules : bool, default False
        Whether to use super-module scores instead of module scores.
    ct_aware : bool, optional
        Whether to use cell-type-aware interacting cell scores. If ``None``, cell-type-aware
        scores are used only when standard scores are absent and matching cell-type-aware
        scores are present.
    """
    MODULE_KEY = "super_module_scores" if use_super_modules else "module_scores"

    if cor_method not in ["pearson", "spearman"]:
        raise ValueError(f'Invalid method: {cor_method}. Choose either "pearson" or "spearman".')

    adata.uns["cor_method"] = cor_method

    if test not in ["parametric", "non-parametric"]:
        raise ValueError('The "test" variable should be one of ["parametric", "non-parametric"].')

    test_str = "p" if test == "parametric" else "np"

    if interaction_type not in ["metabolite", "gene_pair"]:
        raise ValueError(
            'The "interaction_type" variable should be one of ["metabolite", "gene_pair"].'
        )

    interaction_type_str = "m" if interaction_type == "metabolite" else "gp"
    ct_scores_key = f"ct_interacting_cell_results_{test_str}_{interaction_type_str}_cs_df"

    if ct_aware is None:
        ct_aware = "interacting_cell_results" not in adata.uns and ct_scores_key in adata.obsm

    if ct_aware:
        if only_sig_values:
            raise ValueError(
                "only_sig_values=True is not supported for cell-type-aware "
                "interaction-module correlations."
            )
        if ct_scores_key not in adata.obsm:
            raise KeyError(
                f"Missing adata.obsm['{ct_scores_key}']. Run "
                "compute_interacting_cell_scores(mode='cell_type') first."
            )
        interaction_scores = pd.DataFrame(adata.obsm[ct_scores_key], index=adata.obs_names)
    elif only_sig_values:
        sig_str = "FDR" if use_FDR else "pval"
        interaction_scores = adata.uns["interacting_cell_results"][test_str][interaction_type_str][
            f"cs_sig_{sig_str}"
        ]
    else:
        interaction_scores = adata.uns["interacting_cell_results"][test_str][interaction_type_str][
            "cs"
        ]
        interaction_type_names_key = (
            "metabolites" if interaction_type == "metabolite" else "gene_pairs_sig_names"
        )
        interaction_scores = pd.DataFrame(
            interaction_scores,
            index=adata.obs_names,
            columns=adata.uns[interaction_type_names_key],
        )

    if normalize_values:
        interaction_scores = interaction_scores.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
        )  # We apply min-max normalization

    metabolites = interaction_scores.columns.tolist()
    modules = adata.obsm[MODULE_KEY].columns.tolist()

    cor_pval_df = pd.DataFrame(index=modules)
    cor_coef_df = pd.DataFrame(index=modules)

    for metab in metabolites:
        correlation_values = []
        pvals = []

        for module in modules:
            metab_df = interaction_scores[metab]
            module_df = adata.obsm[MODULE_KEY][module]

            if cor_method == "pearson":
                correlation_value, pval = pearsonr(metab_df, module_df)
            elif cor_method == "spearman":
                correlation_value, pval = spearmanr(metab_df, module_df)

            correlation_values.append(correlation_value)
            pvals.append(pval)

        cor_coef_df[metab] = correlation_values
        cor_pval_df[metab] = pvals

    cor_pval_df = cor_pval_df.replace(np.nan, 1)
    cor_coef_df = cor_coef_df.replace(np.nan, 0)

    if cor_pval_df.size == 0:
        import warnings

        warnings.warn(
            "compute_interaction_module_correlation: no metabolite×module pairs to test "
            "(interaction_scores has 0 columns). Storing empty result DataFrames. "
            "Check that compute_interacting_cell_scores produced non-empty scores.",
            stacklevel=2,
        )
        cor_FDR_df = cor_pval_df.copy()
    else:
        cor_FDR_values = multipletests(cor_pval_df.values.flatten(), method="fdr_bh")[1]
        cor_FDR_df = pd.DataFrame(
            cor_FDR_values.reshape(cor_pval_df.shape),
            index=cor_pval_df.index,
            columns=cor_pval_df.columns,
        )

    adata.uns["interaction_module_correlation_coefs"] = cor_coef_df
    adata.uns["interaction_module_correlation_pvals"] = cor_pval_df
    adata.uns["interaction_module_correlation_FDR"] = cor_FDR_df

    return
