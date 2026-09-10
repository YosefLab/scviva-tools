import itertools
import time
import warnings
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from scviva.tools.harreman.hotspot import models
from scviva.tools.harreman.preprocessing.anndata import (
    counts_from_anndata_for_genes,
)
from scviva.utils import resolve_device, stats_dtype

from ._metabolite_scoring import compute_metabolite_cs_ct
from ._stats import compute_max_cs, flatten, z_to_pval_fdr


def _make_hashable(pair: tuple[Any, Any]) -> tuple[Any, Any]:
    """Convert list elements in a pair-like tuple to tuples so it can be hashed/used in a set."""
    return tuple(tuple(x) if isinstance(x, list) else x for x in pair)


def compute_ct_cell_communication(
    adata: AnnData,
    layer_key_p_test: Literal["use_raw"] | str | None = None,
    layer_key_np_test: Literal["use_raw"] | str | None = None,
    model: str = None,
    cell_type_key: str | None = None,
    center_counts_for_np_test: bool | None = False,
    subset_gene_pairs: list | None = None,
    subset_metabolites: list | None = None,
    fix_gp: bool | None = False,
    M: int | None = 1000,
    seed: int | None = 42,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None = "both",
    mean: Literal["algebraic"] | Literal["geometric"] | None = "algebraic",
    check_analytic_null: bool | None = False,
    device: torch.device | str = "auto",
    verbose: bool | None = False,
) -> None:
    """Compute cell type-aware cell-cell communication scores.

    Communication is stratified by interacting cell type pairs and supports parametric and
    non-parametric statistical inference.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Required fields include:
            - `uns["gene_pairs"]`: gene pairs involved in communication.
            - `uns["gene_pairs_per_metabolite"]`: maps metabolites to gene pairs.
            - `uns["gene_pairs_per_ct_pair"]`: gene pairs per cell type pair.
            - `obsp["weights"]`: sparse cell-cell proximity matrix.
            - `obs[cell_type_key]`: categorical cell type annotations.
            - `uns["cell_type_pairs"]`: list of interacting cell type pairs.
            - (Optional) `uns["LR_database"]`: for metabolite/pathway annotation.
    layer_key_p_test : str or "use_raw", optional
        Data layer to use for parametric test.
    layer_key_np_test : str or "use_raw", optional
        Data layer to use for non-parametric test.
    model : str, optional
        Normalization model to use for centering gene expression. Options include "none",
        "normal", "bernoulli", or "danb".
    cell_type_key : str, optional
        Key in `adata.obs` corresponding to cell type annotations. Required if not stored in `uns`.
    center_counts_for_np_test : bool, optional (default: False)
        Whether to center expression counts using the specified model before non-parametric
        testing.
    subset_gene_pairs : list, optional
        Subset of gene pairs to consider. If None, uses all pairs.
    subset_metabolites : list, optional
        Subset of metabolites to include in the analysis.
    fix_gp : bool, optional (default: False)
        If True, keeps gene pair identity fixed during permutation testing, randomizing cell types
        only.
    M : int, optional (default: 1000)
        Number of permutations to use if `permutation_test` is True.
    seed : int, optional (default: 42)
        Random seed for permutation reproducibility.
    test : {'parametric', 'non-parametric', 'both'}, optional (default: 'both')
        Specifies which statistical test(s) to run.
    mean : {'algebraic', 'geometric'}, optional (default: 'algebraic')
        Averaging method for multi-gene modules.
    check_analytic_null : bool, optional (default: False)
        Whether to compute Z-scores and p-values under the null distribution for the permutation
        test.
    device : torch.device, optional
        PyTorch device to run computations on. Defaults to CUDA if available.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
        Results are stored in the following `adata.uns` fields:
            - `ct_ccc_results["p"]`: parametric results per gene pair, metabolite, and cell type.
            - `ct_ccc_results["np"]`: non-parametric scores, empirical p-values, and FDRs.
            - `gene_pair_dict`: dictionary mapping metabolites to relevant gene pairs.
            - `gene_pairs_ind`, `gene_pairs_ind_per_ct_pair`: index-referenced gene pair
              representations.
            - `D`: spatial node degree for each cell per cell type pair.
            - `cells`, `genes`: ordered list of cells and genes used in analysis.
            - (optional) `ct_ccc_results["np"]["analytic_null"]`: permutation null outputs.
    """
    start = time.time()
    device = resolve_device(device)
    if verbose:
        print("Starting cell type-aware cell-cell communication analysis...")

    adata.uns["ct_ccc_results"] = {}

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError(
            'The "test" variable should be one of ["both", "parametric", "non-parametric"].'
        )

    if mean not in ["algebraic", "geometric"]:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')

    if "cell_type_key" in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns["cell_type_key"]
    elif "cell_type_key" not in adata.uns and cell_type_key is None:
        raise ValueError('Please provide the "cell_type_key" argument.')

    adata.uns["layer_key_p_test"] = layer_key_p_test
    adata.uns["layer_key_np_test"] = layer_key_np_test
    adata.uns["model"] = model
    adata.uns["cell_type_key"] = cell_type_key
    adata.uns["center_counts_for_np_test"] = center_counts_for_np_test
    adata.uns["mean"] = mean

    run_ct_cell_communication_analysis(
        adata,
        layer_key_p_test,
        layer_key_np_test,
        model,
        cell_type_key,
        center_counts_for_np_test,
        subset_gene_pairs,
        subset_metabolites,
        fix_gp,
        M,
        seed,
        test,
        mean,
        check_analytic_null,
        device,
        verbose,
    )

    if verbose:
        print("Obtaining cell type-aware communication results...")
    get_ct_cell_communication_results(
        adata,
        adata.uns["genes"],
        adata.uns["cells"],
        layer_key_p_test,
        layer_key_np_test,
        model,
        adata.obs[cell_type_key],
        adata.uns["cell_type_pairs"],
        adata.uns["D"],
        test,
        device,
    )

    if verbose:
        print(
            "Finished computing cell type-aware cell-cell communication analysis in %.3f seconds"
            % (time.time() - start)
        )

    return


def run_ct_cell_communication_analysis(
    adata: AnnData,
    layer_key_p_test: Literal["use_raw"] | str | None,
    layer_key_np_test: Literal["use_raw"] | str | None,
    model: str | None,
    cell_type_key: str | None,
    center_counts_for_np_test: bool | None,
    subset_gene_pairs: list | None,
    subset_metabolites: list | None,
    fix_gp: bool | None,
    M: int | None,
    seed: int | None,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None,
    mean: Literal["algebraic"] | Literal["geometric"] | None,
    check_analytic_null: bool | None,
    device: torch.device | str,
    verbose: bool | None,
) -> None:
    """Run the cell type-aware CCC score and significance workflow."""
    # Cell metadata/obs_names never differ between `adata` and `adata.raw`,
    # and `Raw` has no `.obs` attribute to read them from regardless.
    obs = adata.obs
    cells = adata.obs_names.values.astype(str)

    sample_specific = "sample_key" in adata.uns

    fix_ct = True if adata.uns["fix_ct"] else False

    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    genes = list(np.unique(list(flatten(adata.uns["gene_pairs"]))))
    adata.uns["genes"] = genes

    cell_types = obs[cell_type_key]
    cell_type_pairs = adata.uns.get("cell_type_pairs")
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", {})

    weights = adata.obsp["weights"]

    used_ct_pairs = list({ct for cell_type_pair in cell_type_pairs for ct in cell_type_pair})
    all_cell_types = set(cell_types.unique())
    used_ct_pairs_set = set(used_ct_pairs)
    if used_ct_pairs_set < all_cell_types:
        keep_mask = cell_types[cells].isin(used_ct_pairs).values
        keep_indices = np.where(keep_mask)[0]
        weights = weights[keep_indices][:, keep_indices]
        cells = cells[keep_indices]
        cell_types = cell_types.loc[cells]

    adata.uns["cells"] = cells

    weights_ct_pairs = create_weights_ct_pairs(
        weights.tocoo(), cell_types, cell_type_pairs, device
    )

    row_degrees = torch.sparse.sum(weights_ct_pairs, dim=2).to_dense()
    col_degrees = torch.sparse.sum(weights_ct_pairs, dim=1).to_dense()
    D = row_degrees + col_degrees
    if used_ct_pairs_set < all_cell_types:
        D_full = torch.zeros(
            (len(cell_type_pairs), adata.shape[0]),
            device=weights_ct_pairs.device,
            dtype=weights_ct_pairs.dtype,
        )
        D_full[:, keep_indices] = D
        adata.uns["D"] = D_full.cpu().numpy()
    else:
        adata.uns["D"] = D.cpu().numpy()

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

    # Cell-type pair-specific indices
    gene_pairs_ind_per_ct_pair = defaultdict(list)
    gene_pairs_per_ct_pair_ind = defaultdict(list)
    for ct_pair, gpairs in gene_pairs_per_ct_pair.items():
        for pair in gpairs:
            if pair not in gene_pairs:
                continue
            idx = gene_pairs.index(pair)
            gene_pairs_ind_per_ct_pair[ct_pair].append(gene_pairs_ind[idx])
            gene_pairs_per_ct_pair_ind[ct_pair].append(idx)

    adata.uns["gene_pairs_ind_per_ct_pair"] = dict(gene_pairs_ind_per_ct_pair)
    adata.uns["gene_pairs_per_ct_pair_ind"] = dict(gene_pairs_per_ct_pair_ind)

    gene_pairs_ind_set = {_make_hashable(pair) for pair in gene_pairs_ind}
    ct_specific_gene_pairs = [
        i
        for i, pairs in enumerate(gene_pairs_ind_per_ct_pair.values())
        if {_make_hashable(pair) for pair in pairs} < gene_pairs_ind_set
    ]

    # Metabolite-gene pair preparation
    gp_metab = adata.uns["gene_pairs_per_metabolite"]
    metabolite_gene_pair_df = (
        pd.DataFrame.from_dict(gp_metab, orient="index")
        .rename_axis("metabolite")
        .explode(["gene_pair", "gene_type"])
        .reset_index()
    )

    if "LR_database" in adata.uns:
        merged = metabolite_gene_pair_df.merge(
            adata.uns["LR_database"], left_on="metabolite", right_on="interaction_name", how="left"
        )
        LR_df = merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df.loc[
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite), "metabolite"
        ] = LR_df["pathway_name"].values

    if subset_metabolites:
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df.metabolite.isin(subset_metabolites)
        ]

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

        adata.uns["ct_ccc_results"]["p"] = {"gp": {}, "m": {}}

        weights_sq_data = weights_ct_pairs.values() ** 2
        weights_sq = torch.sparse_coo_tensor(
            weights_ct_pairs.indices(),
            weights_sq_data,
            weights_ct_pairs.shape,
            device=weights_ct_pairs.device,
        )
        Wtot2 = torch.sparse.sum(weights_sq, dim=(1, 2)).to_dense()

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

        counts_1 = standardize_ct_counts(
            adata, counts_1, model, num_umi, sample_specific, cell_types
        )
        counts_2 = standardize_ct_counts(
            adata, counts_2, model, num_umi, sample_specific, cell_types
        )

        # Compute CCC scores
        cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
        for ct_pair in range(len(cell_type_pairs)):
            W = weights_ct_pairs[ct_pair].coalesce()
            WX2t = torch.sparse.mm(W, counts_2.T)
            cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
        adata.uns["ct_ccc_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs_ct(
            cs_gp,
            cell_type_key,
            gene_pair_dict,
            gene_pairs_per_ct_pair_ind,
            ct_specific_gene_pairs,
            interacting_cell_scores=False,
        )
        adata.uns["ct_ccc_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        EG2_gp = torch.zeros_like(cs_gp) if fix_ct or fix_gp else Wtot2
        if fix_ct:
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                W_sq_data = W.values() ** 2
                W_sq = torch.sparse_coo_tensor(W.indices(), W_sq_data, W.shape, device=W.device)
                X1_sq = counts_1**2
                EG2_gp[ct_pair] = torch.sparse.mm(W_sq, X1_sq.T).sum(0)
        elif fix_gp:
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                W_sq_data = W.values() ** 2
                W_sq = torch.sparse_coo_tensor(W.indices(), W_sq_data, W.shape, device=W.device)
                X1_sq = counts_1**2
                X2_sq = counts_2**2
                EG2_gp[ct_pair] = (X1_sq.T * torch.sparse.mm(W_sq, X2_sq.T)).sum(0)

        Z_gp, Z_m = compute_ct_p_results(
            cs_gp,
            cs_m,
            gene_pairs_per_ct_pair_ind,
            ct_specific_gene_pairs,
            EG2_gp,
            cell_type_key,
            gene_pair_dict,
        )

        # Convert tensors to numpy for statsmodels and pandas
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        # Compute p-values and FDRs
        Z_pvals_gp, FDR_gp = z_to_pval_fdr(Z_gp_np)
        Z_pvals_m, FDR_m = z_to_pval_fdr(Z_m_np)

        # Store in AnnData
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z"] = Z_gp_np
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z_pval"] = Z_pvals_gp
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z_FDR"] = FDR_gp
        adata.uns["ct_ccc_results"]["p"]["m"]["Z"] = Z_m_np
        adata.uns["ct_ccc_results"]["p"]["m"]["Z_pval"] = Z_pvals_m
        adata.uns["ct_ccc_results"]["p"]["m"]["Z_FDR"] = FDR_m

    if test in ["non-parametric", "both"]:
        if verbose:
            print("Running the non-parametric test...")

        adata.uns["ct_ccc_results"]["np"] = {"gp": {}, "m": {}}

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
            counts_1 = standardize_ct_counts(
                adata, counts_1, model, num_umi, sample_specific, cell_types
            )
            counts_2 = standardize_ct_counts(
                adata, counts_2, model, num_umi, sample_specific, cell_types
            )

        if center_counts_for_np_test and test == "both":
            adata.uns["ct_ccc_results"]["np"]["gp"]["cs"] = np.array(
                adata.uns["ct_ccc_results"]["p"]["gp"]["cs"]
            )
            adata.uns["ct_ccc_results"]["np"]["m"]["cs"] = np.array(
                adata.uns["ct_ccc_results"]["p"]["m"]["cs"]
            )
        else:
            cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, counts_2.T)
                cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
            adata.uns["ct_ccc_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs_ct(
                cs_gp,
                cell_type_key,
                gene_pair_dict,
                gene_pairs_per_ct_pair_ind,
                ct_specific_gene_pairs,
                interacting_cell_scores=False,
            )
            adata.uns["ct_ccc_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        perm_cs_gp = torch.zeros(
            (len(cell_type_pairs), counts_1.shape[0], M), dtype=stats_dtype(device), device=device
        )
        perm_cs_m = torch.zeros(
            (len(cell_type_pairs), len(gene_pair_dict), M),
            dtype=stats_dtype(device),
            device=device,
        )

        if check_analytic_null:
            gp_zs_perm_array = torch.zeros_like(perm_cs_gp)
            gp_pvals_perm_array = torch.zeros_like(perm_cs_gp)
            m_zs_perm_array = torch.zeros_like(perm_cs_m)
            m_pvals_perm_array = torch.zeros_like(perm_cs_m)

        if fix_gp:
            c1_perm = counts_1
            c2_perm = counts_2

        torch.manual_seed(seed)
        for i in tqdm(range(M), desc="Permutation test"):
            if fix_gp:
                indices = torch.randperm(len(cell_types)).numpy()
                shuffled_cell_types = cell_types.iloc[indices].reset_index(drop=True)
                weights_ct_pairs = create_weights_ct_pairs(
                    weights.tocoo(), shuffled_cell_types, cell_type_pairs, device
                )
            else:
                cell_type_labels = torch.tensor(
                    cell_types.astype("category").cat.codes.values, device=counts_1.device
                )
                idx = torch.empty_like(cell_type_labels, dtype=torch.int64)

                for ct in torch.unique(cell_type_labels):
                    ct_mask = cell_type_labels == ct
                    ct_indices = torch.nonzero(ct_mask, as_tuple=True)[0]
                    permuted_indices = ct_indices[torch.randperm(len(ct_indices))]
                    idx[ct_indices] = permuted_indices

                c1_perm = counts_1 if fix_ct else counts_1[:, idx.long()]
                c2_perm = counts_2[:, idx.long()]

            cs_gp = torch.zeros((len(cell_type_pairs), c1_perm.shape[0]), device=c1_perm.device)
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, c2_perm.T)
                cs_gp[ct_pair] = (c1_perm.T * WX2t).sum(0)
            perm_cs_gp[:, :, i] = cs_gp

            cs_m = compute_metabolite_cs_ct(
                cs_gp,
                cell_type_key,
                gene_pair_dict,
                gene_pairs_per_ct_pair_ind,
                ct_specific_gene_pairs,
                interacting_cell_scores=False,
            )
            perm_cs_m[:, :, i] = cs_m

            if check_analytic_null:
                Z_gp_perm, Z_m_perm = compute_ct_p_results(
                    cs_gp,
                    cs_m,
                    gene_pairs_per_ct_pair_ind,
                    ct_specific_gene_pairs,
                    EG2_gp,
                    cell_type_key,
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

        adata.uns["ct_ccc_results"]["np"]["gp"]["perm_cs"] = perm_cs_gp.detach().cpu().numpy()
        adata.uns["ct_ccc_results"]["np"]["m"]["perm_cs"] = perm_cs_m.detach().cpu().numpy()

        x_gp = np.sum(
            adata.uns["ct_ccc_results"]["np"]["gp"]["perm_cs"]
            > adata.uns["ct_ccc_results"]["np"]["gp"]["cs"][:, :, np.newaxis],
            axis=2,
        )
        x_m = np.sum(
            adata.uns["ct_ccc_results"]["np"]["m"]["perm_cs"]
            > adata.uns["ct_ccc_results"]["np"]["m"]["cs"][:, :, np.newaxis],
            axis=2,
        )

        pvals_gp = (x_gp + 1) / (M + 1)
        pvals_m = (x_m + 1) / (M + 1)

        adata.uns["ct_ccc_results"]["np"]["gp"]["pval"] = pvals_gp
        adata.uns["ct_ccc_results"]["np"]["gp"]["FDR"] = multipletests(
            pvals_gp.flatten(), method="fdr_bh"
        )[1].reshape(pvals_gp.shape)
        adata.uns["ct_ccc_results"]["np"]["m"]["pval"] = pvals_m
        adata.uns["ct_ccc_results"]["np"]["m"]["FDR"] = multipletests(
            pvals_m.flatten(), method="fdr_bh"
        )[1].reshape(pvals_m.shape)

        if check_analytic_null:
            adata.uns["ct_ccc_results"]["np"]["analytic_null"] = {
                "gp_zs_perm": gp_zs_perm_array.detach().cpu().numpy(),
                "gp_pvals_perm": gp_pvals_perm_array.detach().cpu().numpy(),
                "m_zs_perm": m_zs_perm_array.detach().cpu().numpy(),
                "m_pvals_perm": m_pvals_perm_array.detach().cpu().numpy(),
            }

    adata.uns["cell_types"] = cell_types.tolist() if cell_type_key else None

    if verbose:
        print("Non-parametric test finished.")

    return


def standardize_ct_counts(
    adata: AnnData,
    counts: torch.Tensor,
    model: str | None,
    num_umi: torch.Tensor,
    sample_specific: bool,
    cell_types: pd.Series,
) -> torch.Tensor:
    """Standardize counts within cell types and optional samples."""
    if sample_specific:
        sample_key = adata.uns["sample_key"]
        for sample in adata.obs[sample_key].unique():
            subset = np.where(adata.obs[sample_key] == sample)[0]
            counts[:, subset] = center_ct_counts_torch(
                counts[:, subset], num_umi[subset], model, cell_types[subset]
            )
    else:
        counts = center_ct_counts_torch(counts, num_umi, model, cell_types)

    return counts


def create_weights_ct_pairs(
    weights: Any,
    cell_types: pd.Series,
    cell_type_pairs: list[tuple[str, str]],
    device: torch.device | str,
) -> torch.Tensor:
    """Create sparse weight tensors for each cell type pair."""
    indices = torch.tensor([weights.row, weights.col], dtype=torch.long, device=device)
    values = torch.tensor(weights.data, dtype=stats_dtype(device), device=device)
    shape = weights.shape

    cell_type_cats = cell_types.astype("category")
    cell_type_codes = torch.tensor(
        cell_type_cats.cat.codes.values, dtype=torch.long, device=device
    )
    ct_name_to_code = {name: code for code, name in enumerate(cell_type_cats.cat.categories)}

    row_idx, col_idx = indices
    sender_types = cell_type_codes[row_idx]
    receiver_types = cell_type_codes[col_idx]

    weights_list = []
    coord_list = []

    for i, (ct1, ct2) in enumerate(cell_type_pairs):
        code1 = ct_name_to_code[ct1]
        code2 = ct_name_to_code[ct2]

        pair_mask = (sender_types == code1) & (receiver_types == code2)
        if pair_mask.sum() == 0:
            continue

        pair_values = values[pair_mask]
        pair_coords = torch.stack(
            [
                torch.full((pair_values.shape[0],), i, dtype=torch.long, device=device),
                row_idx[pair_mask],
                col_idx[pair_mask],
            ],
            dim=0,
        )

        weights_list.append(pair_values)
        coord_list.append(pair_coords)

    all_values = torch.cat(weights_list)
    all_coords = torch.cat(coord_list, dim=1)
    weights_ct_pairs = torch.sparse_coo_tensor(
        all_coords, all_values, (len(cell_type_pairs), shape[0], shape[1]), device=device
    )
    weights_ct_pairs = weights_ct_pairs.coalesce()

    return weights_ct_pairs


def compute_ct_interacting_cell_scores(
    adata: str | AnnData,
    center_counts_for_np_test: bool | None = False,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None = "both",
    restrict_significance: Literal["gene pairs"]
    | Literal["metabolites"]
    | Literal["both"]
    | None = "both",
    device: torch.device | str = "auto",
    verbose: bool | None = False,
) -> None:
    """Compute cell-type-aware interacting cell scores for gene pairs and metabolites.

    Parameters
    ----------
    adata : AnnData or str
        Must contain:
        - ``uns['model']``, ``uns['mean']``
        - ``uns['cell_type_key']`` and ``obs[cell_type_key]`` for cell types
        - ``uns['gene_pairs']``, ``uns['gene_pairs_per_ct_pair']``
        - ``uns['gene_pairs_per_metabolite']``
        - ``uns['ct_ccc_results']`` with significance information
        - ``obsp['weights']`` (spatial proximity matrix)
    center_counts_for_np_test : bool, default False
        Whether to standardize counts before the non-parametric test.
    test : {"parametric", "non-parametric", "both"}
        Which statistical test(s) to run.
    restrict_significance : {"gene pairs", "metabolites", "both"}
        Only use cell-type-pair interactions that were significant in cell-type-aware CCC results.
    device : torch.device
        CPU or GPU device for PyTorch computations.
    verbose : bool, default False
        Print detailed progress messages.
    """
    start = time.time()
    device = resolve_device(device)
    if verbose:
        print("Computing cell type-aware gene pair and metabolite scores...")

    adata.uns["ct_interacting_cell_results"] = {}

    model = adata.uns["model"]
    mean = adata.uns["mean"]

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError(
            'The "test" variable should be one of ["both", "parametric", "non-parametric"].'
        )

    if restrict_significance not in ["both", "gene pairs", "metabolites"]:
        raise ValueError(
            'The "restrict_significance" variable should be one of '
            '["both", "gene pairs", "metabolites"].'
        )

    sample_specific = "sample_key" in adata.uns

    layer_key_p_test = adata.uns.get("layer_key_p_test", None)
    layer_key_np_test = adata.uns.get("layer_key_np_test", None)

    # Cell metadata/obs_names never differ between `adata` and `adata.raw`,
    # and `Raw` has no `.obs` attribute to read them from regardless.
    obs = adata.obs
    cells = adata.obs_names.values.astype(str)

    gene_pairs = adata.uns.get("gene_pairs", None)
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", None)

    gp_metab = adata.uns["gene_pairs_per_metabolite"]
    metabolite_gene_pair_df = (
        pd.DataFrame.from_dict(gp_metab, orient="index")
        .rename_axis("metabolite")
        .explode(["gene_pair", "gene_type"])
        .reset_index()
    )

    if "LR_database" in adata.uns:
        merged = metabolite_gene_pair_df.merge(
            adata.uns["LR_database"], left_on="metabolite", right_on="interaction_name", how="left"
        )
        LR_df = merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df.loc[
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite), "metabolite"
        ] = LR_df["pathway_name"].values

    cell_type_pairs = adata.uns.get("cell_type_pairs")
    cell_type_pairs = [tuple(x) for x in cell_type_pairs]

    cell_com_gp_df = adata.uns["ct_ccc_results"]["cell_com_df_gp_sig"].copy()
    if restrict_significance in ["both", "gene pairs"]:
        ct_pairs_gp_set = {tuple(x) for x in cell_com_gp_df[["Cell Type 1", "Cell Type 2"]].values}
        cell_type_pairs = [ct_pair for ct_pair in cell_type_pairs if ct_pair in ct_pairs_gp_set]

        cell_com_gp_df[["Gene 1", "Gene 2"]] = cell_com_gp_df[["Gene 1", "Gene 2"]].map(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

        gene_pairs_set = {tuple(x) for x in cell_com_gp_df[["Gene 1", "Gene 2"]].values}
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df["gene_pair"].isin(gene_pairs_set)
        ]

    cell_com_m_df = adata.uns["ct_ccc_results"]["cell_com_df_m_sig"].copy()
    if restrict_significance in ["both", "metabolites"]:
        ct_pairs_m_set = {tuple(x) for x in cell_com_m_df[["Cell Type 1", "Cell Type 2"]].values}
        missing_ct_pairs = [
            ct_pair for ct_pair in ct_pairs_m_set if ct_pair not in cell_type_pairs
        ]
        if len(missing_ct_pairs) > 0:
            warnings.warn(
                "The following cell type pairs are not included in the "
                f'"cell_type_pairs" set: {missing_ct_pairs}',
                stacklevel=2,
            )

        metabolite_set = set(cell_com_m_df["metabolite"].values)
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df["metabolite"].isin(metabolite_set)
        ]

    if metabolite_gene_pair_df.empty:
        if restrict_significance == "both":
            raise ValueError(
                "There are no significant gene pairs that belong to a significant metabolite."
            )
        if restrict_significance == "gene pairs":
            raise ValueError("There are no significant gene pairs.")
        if restrict_significance == "metabolites":
            raise ValueError("There are no significant metabolites.")

    genes = adata.uns["genes"]
    gene_pairs_sig = []
    if gene_pairs:
        for g1, g2 in gene_pairs:
            g1 = tuple(g1) if isinstance(g1, list) else g1
            g2 = tuple(g2) if isinstance(g2, list) else g2
            if not metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"] == (g1, g2)].empty:
                gene_pairs_sig.append((g1, g2))

    gene_pairs_sig_ind = []
    for pair in gene_pairs_sig:
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
        gene_pairs_sig_ind.append((idx1, idx2))

    cell_type_key = adata.uns.get("cell_type_key")
    cell_types = obs[cell_type_key]
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", {})

    weights = adata.obsp["weights"]

    used_ct_pairs = list({ct for cell_type_pair in cell_type_pairs for ct in cell_type_pair})
    all_cell_types = set(cell_types.unique())
    used_ct_pairs_set = set(used_ct_pairs)
    if used_ct_pairs_set < all_cell_types:
        keep_mask = cell_types[cells].isin(used_ct_pairs).values
        keep_indices = np.where(keep_mask)[0]
        weights = weights[keep_indices][:, keep_indices]
        cells = cells[keep_indices]
        cell_types = cell_types.loc[
            cells
        ]  # Eventually only keep the cell type pairs with at least one significant gene pair

    weights_ct_pairs = create_weights_ct_pairs(
        weights.tocoo(), cell_types, cell_type_pairs, device
    )

    gene_pairs_per_ct_pair_sig = {}
    for ct_pair in gene_pairs_per_ct_pair.keys():
        if ct_pair not in cell_type_pairs:
            continue
        cell_com_df_ct_pair = cell_com_gp_df[
            (cell_com_gp_df["Cell Type 1"] == ct_pair[0])
            & (cell_com_gp_df["Cell Type 2"] == ct_pair[1])
        ]
        gene_pairs_per_ct_pair_sig[ct_pair] = [
            tuple(x) for x in cell_com_df_ct_pair[["Gene 1", "Gene 2"]].values
        ]

    # Cell-type pair-specific indices
    gene_pairs_ind_per_ct_pair_sig = defaultdict(list)
    gene_pairs_per_ct_pair_sig_ind = defaultdict(list)
    for ct_pair, gpairs in gene_pairs_per_ct_pair_sig.items():
        for pair in gpairs:
            if pair not in gene_pairs_sig:
                continue
            idx = gene_pairs_sig.index(pair)
            gene_pairs_ind_per_ct_pair_sig[ct_pair].append(gene_pairs_sig_ind[idx])
            gene_pairs_per_ct_pair_sig_ind[ct_pair].append(idx)

    adata.uns["gene_pairs_ind_per_ct_pair_sig"] = dict(gene_pairs_ind_per_ct_pair_sig)
    adata.uns["gene_pairs_per_ct_pair_sig_ind"] = dict(gene_pairs_per_ct_pair_sig_ind)

    gene_pairs_sig_ind_set = {_make_hashable(pair) for pair in gene_pairs_sig_ind}
    ct_specific_gene_pairs = [
        i
        for i, pairs in enumerate(gene_pairs_ind_per_ct_pair_sig.values())
        if {_make_hashable(pair) for pair in pairs} < gene_pairs_sig_ind_set
    ]

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

    gene_pair_to_metabolite_indices = defaultdict(set)
    for met_idx, met in enumerate(metabolites):
        for gene_pair_idx in gene_pair_dict.get(met, []):
            gene_pair_to_metabolite_indices[gene_pair_idx].add(met_idx)

    ct_pair_to_metabolite_indices = {}
    for ct_pair, gene_pair_indices in gene_pairs_per_ct_pair_sig_ind.items():
        met_indices = set()
        for gi in gene_pair_indices:
            met_indices.update(gene_pair_to_metabolite_indices.get(gi, []))
        ct_pair_to_metabolite_indices[ct_pair] = sorted(met_indices)

    def concat_tuple_elements(t, sep="_"):
        return sep.join(
            s for item in t for s in (item if isinstance(item, list | tuple) else [item])
        )

    if test in ["parametric", "both"]:
        if verbose:
            print("Running the parametric test...")

        adata.uns["ct_interacting_cell_results"]["p"] = {"gp": {}, "m": {}}

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

        counts_1 = standardize_ct_counts(
            adata[cells, :], counts_1, model, num_umi, sample_specific, cell_types
        )
        counts_2 = standardize_ct_counts(
            adata[cells, :], counts_2, model, num_umi, sample_specific, cell_types
        )

        cs_gp = torch.zeros(
            (len(cell_type_pairs), counts_1.shape[1], counts_1.shape[0]), device=counts_1.device
        )
        for ct_pair in range(len(cell_type_pairs)):
            W = weights_ct_pairs[ct_pair].coalesce()
            WX2t = torch.sparse.mm(W, counts_2.T)
            cs_gp[ct_pair] = counts_1.T * WX2t
        adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs_ct(
            cs_gp,
            cell_type_key,
            gene_pair_dict,
            gene_pairs_per_ct_pair_sig_ind,
            ct_specific_gene_pairs,
            interacting_cell_scores=True,
        )
        adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        column_names = []
        scores = []
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_sig_ind.keys()):
            gp_list = gene_pairs_per_ct_pair_sig_ind[ct_pair]
            for gp in gp_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"][i, :, gp] == 0
                ):
                    continue
                column_names.append(
                    f"{' - '.join(ct_pair)}: {concat_tuple_elements(gene_pairs_sig[gp])}"
                )
                scores.append(adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"][i, :, gp])

        cs_gp_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_gp_df = cs_gp_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_p_gp_cs_df"] = cs_gp_df

        column_names = []
        scores = []
        for i, ct_pair in enumerate(ct_pair_to_metabolite_indices.keys()):
            metab_list = ct_pair_to_metabolite_indices[ct_pair]
            for metab in metab_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"][i, :, metab] == 0
                ):
                    continue
                column_names.append(f"{' - '.join(ct_pair)}: {metabolites[metab]}")
                scores.append(
                    adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"][i, :, metab]
                )

        cs_m_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_m_df = cs_m_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_p_m_cs_df"] = cs_m_df

        if verbose:
            print("Parametric test finished.")

    if test in ["non-parametric", "both"]:
        if verbose:
            print("Running the non-parametric test...")

        adata.uns["ct_interacting_cell_results"]["np"] = {"gp": {}, "m": {}}

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
            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        if center_counts_for_np_test and test == "both":
            adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"] = np.array(
                adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"]
            )
            adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"] = np.array(
                adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"]
            )
        else:
            cs_gp = torch.zeros(
                (len(cell_type_pairs), counts_1.shape[1], counts_1.shape[0]),
                device=counts_1.device,
            )
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, counts_2.T)
                cs_gp[ct_pair] = counts_1.T * WX2t
            adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"] = (
                cs_gp.detach().cpu().numpy()
            )
            cs_m = compute_metabolite_cs_ct(
                cs_gp,
                cell_type_key,
                gene_pair_dict,
                gene_pairs_per_ct_pair_sig_ind,
                ct_specific_gene_pairs,
                interacting_cell_scores=True,
            )
            adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        column_names = []
        scores = []
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_sig_ind.keys()):
            gp_list = gene_pairs_per_ct_pair_sig_ind[ct_pair]
            for gp in gp_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"][i, :, gp] == 0
                ):
                    continue
                column_names.append(
                    f"{' - '.join(ct_pair)}: {concat_tuple_elements(gene_pairs_sig[gp])}"
                )
                scores.append(adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"][i, :, gp])

        cs_gp_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_gp_df = cs_gp_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_np_gp_cs_df"] = cs_gp_df

        column_names = []
        scores = []
        for i, ct_pair in enumerate(ct_pair_to_metabolite_indices.keys()):
            metab_list = ct_pair_to_metabolite_indices[ct_pair]
            for metab in metab_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"][i, :, metab] == 0
                ):
                    continue
                column_names.append(f"{' - '.join(ct_pair)}: {metabolites[metab]}")
                scores.append(
                    adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"][i, :, metab]
                )

        cs_m_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_m_df = cs_m_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_np_m_cs_df"] = cs_m_df

        if verbose:
            print("Non-parametric test finished.")

    if verbose:
        print(
            "Finished computing cell type-aware gene pair and metabolite scores in %.3f seconds"
            % (time.time() - start)
        )

    return


def compute_ct_p_results(
    C_gp: torch.Tensor,
    C_m: torch.Tensor,
    gene_pairs_per_ct_pair_ind: dict[Any, list[int]],
    ct_specific_gene_pairs: list[int],
    EG2_gp: torch.Tensor,
    cell_type_key: str | None,
    gene_pair_dict: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cell type-aware parametric Z-scores."""
    EG2_gp = EG2_gp.unsqueeze(1).expand(-1, C_gp.shape[1]) if len(EG2_gp.shape) == 1 else EG2_gp

    stdG = torch.sqrt(EG2_gp)
    stdG[stdG == 0] = 1

    Z_gp = C_gp / stdG

    EG2_m = compute_metabolite_cs_ct(
        EG2_gp,
        cell_type_key,
        gene_pair_dict,
        gene_pairs_per_ct_pair_ind,
        ct_specific_gene_pairs,
        interacting_cell_scores=False,
    )
    if not isinstance(EG2_m, torch.Tensor):
        device = EG2_gp.device
        EG2_m = torch.tensor(EG2_m, device=device, dtype=stats_dtype(device))

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    Z_m = C_m / stdG_m

    return Z_gp, Z_m


def get_ct_cell_communication_results(
    adata: AnnData,
    genes: list[str],
    cells: np.ndarray,
    layer_key_p_test: Literal["use_raw"] | str | None,
    layer_key_np_test: Literal["use_raw"] | str | None,
    model: str | None,
    cell_types: pd.Series,
    cell_type_pairs: list[tuple[str, str]],
    D: np.ndarray | torch.Tensor,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None,
    device: torch.device | str,
) -> None:
    """Assemble cell type-aware communication result dataframes."""
    gene_pairs_ind_per_ct_pair = adata.uns["gene_pairs_ind_per_ct_pair"]
    gene_pair_dict = adata.uns["gene_pair_dict"]
    genes = adata.uns["genes"]

    sample_specific = "sample_key" in adata.uns

    if isinstance(D, np.ndarray):
        D = torch.tensor(D, dtype=stats_dtype(device), device=device)

    def idx_to_gene(idx):
        return [genes[i] for i in idx] if isinstance(idx, list) else genes[idx]

    records = [
        {
            "Cell Type 1": ct1,
            "Cell Type 2": ct2,
            "Gene 1": idx_to_gene(gp[0]),
            "Gene 2": idx_to_gene(gp[1]),
        }
        for (ct1, ct2), gp_list in gene_pairs_ind_per_ct_pair.items()
        for gp in gp_list
    ]
    cell_com_df_gp = pd.DataFrame.from_records(records)

    # Generate metabolite interaction table
    ct_pairs = list(gene_pairs_ind_per_ct_pair.keys())
    metabolites = list(gene_pair_dict.keys())
    cell_com_df_m = pd.DataFrame(
        [
            {"Cell Type 1": ct1, "Cell Type 2": ct2, "metabolite": m}
            for (ct1, ct2), m in itertools.product(ct_pairs, metabolites)
        ]
    )

    if test in ["parametric", "both"]:
        suffix = "p"
        # Gene pair
        c_values = adata.uns["ct_ccc_results"][suffix]["gp"]["cs"]
        z_values = adata.uns["ct_ccc_results"][suffix]["gp"]["Z"]
        p_values = adata.uns["ct_ccc_results"][suffix]["gp"]["Z_pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["gp"]["Z_FDR"]
        cell_com_df_gp[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_gp["Z"] = z_values.flatten()
        cell_com_df_gp["Z_pval"] = p_values.flatten()
        cell_com_df_gp["Z_FDR"] = fdr_values.flatten()

        counts = counts_from_anndata_for_genes(adata, genes, layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)
        num_umi = counts.sum(dim=0)
        counts_std = standardize_ct_counts(
            adata, counts, model, num_umi, sample_specific, cell_types
        )

        c_values_norm = normalize_ct_values(
            counts_std, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D
        )
        adata.uns["ct_ccc_results"][suffix]["gp"]["cs_norm"] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f"C_norm_{suffix}"] = c_values_norm.cpu().numpy().flatten()

        # Metabolite
        c_values = adata.uns["ct_ccc_results"][suffix]["m"]["cs"]
        z_values = adata.uns["ct_ccc_results"][suffix]["m"]["Z"]
        p_values = adata.uns["ct_ccc_results"][suffix]["m"]["Z_pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["m"]["Z_FDR"]
        cell_com_df_m[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_m["Z"] = z_values.flatten()
        cell_com_df_m["Z_pval"] = p_values.flatten()
        cell_com_df_m["Z_FDR"] = fdr_values.flatten()

    if test in ["non-parametric", "both"]:
        suffix = "np"
        # Gene pair
        c_values = adata.uns["ct_ccc_results"][suffix]["gp"]["cs"]
        p_values = adata.uns["ct_ccc_results"][suffix]["gp"]["pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["gp"]["FDR"]
        cell_com_df_gp[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_gp[f"pval_{suffix}"] = p_values.flatten()
        cell_com_df_gp[f"FDR_{suffix}"] = fdr_values.flatten()

        counts = counts_from_anndata_for_genes(adata, genes, layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=stats_dtype(device), device=device)
        if adata.uns.get("center_counts_for_np_test", False):
            num_umi = counts.sum(dim=0)
            counts = standardize_ct_counts(
                adata, counts, model, num_umi, sample_specific, cell_types
            )

        c_values_norm = normalize_ct_values(
            counts, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D
        )
        adata.uns["ct_ccc_results"][suffix]["gp"]["cs_norm"] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f"C_norm_{suffix}"] = c_values_norm.cpu().numpy().flatten()

        # Metabolite
        c_values = adata.uns["ct_ccc_results"][suffix]["m"]["cs"]
        p_values = adata.uns["ct_ccc_results"][suffix]["m"]["pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["m"]["FDR"]
        cell_com_df_m[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_m[f"pval_{suffix}"] = p_values.flatten()
        cell_com_df_m[f"FDR_{suffix}"] = fdr_values.flatten()

    adata.uns["ct_ccc_results"]["cell_com_df_gp"] = cell_com_df_gp
    adata.uns["ct_ccc_results"]["cell_com_df_m"] = cell_com_df_m

    return


def normalize_ct_values(
    counts: torch.Tensor,
    cell_types: pd.Series | np.ndarray,
    cell_type_pairs: list[tuple[str, str]],
    gene_pairs_per_ct_pair_ind: dict[Any, list[tuple[Any, Any]]],
    lcs: torch.Tensor | np.ndarray,
    D: torch.Tensor,
) -> torch.Tensor:
    """Normalize communication scores within each cell type pair."""
    if isinstance(cell_types, pd.Series):
        cell_types = cell_types.values

    if isinstance(lcs, np.ndarray):
        lcs = torch.tensor(lcs, dtype=counts.dtype, device=counts.device)

    c_values_norm = torch.empty_like(lcs, dtype=counts.dtype, device=counts.device)

    for i, ct_pair in enumerate(cell_type_pairs):
        ct_t, _ = ct_pair

        ct_mask = cell_types == ct_t
        if isinstance(ct_mask, np.ndarray):
            ct_mask = torch.tensor(ct_mask, device=counts.device)

        counts_ct = counts[:, ct_mask]
        D_ct = D[i][ct_mask]
        gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

        lc_maxs = compute_max_cs(D_ct, counts_ct, gene_pairs_ind)
        lc_maxs = torch.where(lc_maxs == 0, torch.tensor(1.0, device=counts.device), lc_maxs)

        c_values = lcs[i] if lcs.ndim == 2 else lcs[i : i + 1]  # allow 1D or 2D lcs
        c_values_norm[i] = c_values / lc_maxs
        c_values_norm[i] = torch.where(
            torch.isinf(c_values_norm[i]),
            torch.tensor(1.0, device=counts.device),
            c_values_norm[i],
        )

    return c_values_norm


def center_ct_counts_torch(
    counts: torch.Tensor, num_umi: torch.Tensor, model: str, cell_types: pd.Series
) -> torch.Tensor:
    """Center counts within cell types.

    counts: Tensor [genes, cells]
    num_umi: Tensor [cells]
    model: 'bernoulli', 'danb', 'normal', or 'none'

    Returns
    -------
        Centered counts within cell types: Tensor [genes, cells]
    """
    # Binarize if using Bernoulli
    if model == "bernoulli":
        counts = (counts > 0).to(stats_dtype(counts.device))
        mu, var, _ = models.apply_model_per_cell_type(
            models.bernoulli_model_torch, counts, num_umi, cell_types
        )
    elif model == "danb":
        mu, var, _ = models.apply_model_per_cell_type(
            models.danb_model_torch, counts, num_umi, cell_types
        )
    elif model == "normal":
        mu, var, _ = models.apply_model_per_cell_type(
            models.normal_model_torch, counts, num_umi, cell_types
        )
    elif model == "none":
        mu, var, _ = models.apply_model_per_cell_type(
            models.none_model_torch, counts, num_umi, cell_types
        )
    else:
        raise ValueError(f"Unsupported model type: {model}")

    # Avoid division by zero
    std = torch.sqrt(var)
    std[std == 0] = 1.0

    centered = (counts - mu) / std
    centered[centered == 0] = 0  # Optional: to match old behavior

    return centered
