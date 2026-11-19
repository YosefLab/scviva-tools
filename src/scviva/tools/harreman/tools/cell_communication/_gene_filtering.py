"""Gene filtering and gene-pair discovery for cell-cell communication analysis."""

import ast
import itertools
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from numba import njit
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from scviva.tools.harreman.preprocessing.anndata import counts_from_anndata_for_genes


def apply_gene_filtering(
    adata: AnnData,
    layer_key: Literal["use_raw"] | str | None = None,
    cell_type_key: str | None = None,
    model: str | None = None,
    feature_elimination: bool | None = False,
    threshold: float | None = 0.2,
    autocorrelation_filt: bool | None = False,
    expression_filt: bool | None = False,
    de_filt: bool | None = False,
    umi_counts_obs_key: str | None = None,
    device: torch.device | str = "auto",
    verbose: bool | None = False,
) -> None:
    """
    Applies multi-step gene filtering to an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (AnnData).
    layer_key : str, optional
        Key to use from `adata.layers` or `"use_raw"` to use `adata.raw.X`.
    cell_type_key : str, optional
        Key in `adata.obs` containing cell type annotations.
    model : str, optional
        Model name for autocorrelation computation.
    feature_elimination : bool, optional (default: False)
        If True, filters genes based on sparsity across all cells.
    threshold : float, optional (default: 0.2)
        Minimum fraction of cells in which the gene must be expressed.
    autocorrelation_filt : str, optional (default: False)
        If True, filters genes based on spatial autocorrelation significance.
    expression_filt : str, optional (default: False)
        If True, filters genes based on expression in each cell type.
    de_filt : str, optional (default: False)
        If True, filters genes based on differential expression between each cell type and the
        rest.
    umi_counts_obs_key : str, optional
        Key in `adata.obs` with total UMI counts per cell. If `None`, inferred from the
        expression matrix.
    device : torch.device, optional
        Device to use for computation (e.g., CUDA or CPU). Defaults to GPU if available.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
    """
    start = time.time()
    if verbose:
        print("Applying gene filtering...")

    adata.uns["autocorrelation_filt"] = autocorrelation_filt
    adata.uns["expression_filt"] = expression_filt
    adata.uns["de_filt"] = de_filt

    db_key = adata.uns["database_varm_key"]

    if feature_elimination:
        perform_feature_elimination(adata, layer_key, db_key, threshold)

    from scviva.tools.harreman.hotspot.local_autocorrelation import compute_local_autocorrelation

    if autocorrelation_filt:
        compute_local_autocorrelation(
            adata=adata,
            layer_key=layer_key,
            database_varm_key=db_key,
            model=model,
            umi_counts_obs_key=umi_counts_obs_key,
            device=device,
            verbose=verbose,
        )

    if expression_filt or de_filt:
        if cell_type_key is None:
            cell_type_key = adata.uns.get("cell_type_key")
            if cell_type_key is None:
                raise ValueError('The "cell_type_key" argument needs to be provided.')

        filtered_genes, filtered_genes_ct = filter_genes(
            adata,
            layer_key,
            db_key,
            cell_type_key,
            expression_filt,
            de_filt,
            autocorrelation_filt,
            threshold=threshold,
        )
        adata.uns["filtered_genes"] = filtered_genes
        adata.uns["filtered_genes_ct"] = filtered_genes_ct

    if verbose:
        print("Finished applying gene filtering in %.3f seconds" % (time.time() - start))

    return


def perform_feature_elimination(
    adata: AnnData,
    layer_key: Literal["use_raw"] | str | None,
    database_varm_key: str,
    threshold: float,
) -> None:
    """
    Filters out genes that are too sparse across all cells.

    Parameters
    ----------
    adata
        Annotated data object (AnnData).
    layer_key
        Which layer to use (or "use_raw").
    database_varm_key
        Key in `adata.varm` pointing to relevant features to filter.
    threshold
        Minimum fraction of cells in which the gene must be expressed.
    """
    use_raw = layer_key == "use_raw"
    store = adata.raw.varm if use_raw else adata.varm
    var_names = adata.raw.var_names if use_raw else adata.var_names

    metab_matrix = store[database_varm_key]
    genes = metab_matrix.loc[(metab_matrix != 0).any(axis=1)].index

    counts = counts_from_anndata_for_genes(adata, genes, layer_key, dense=True)

    valid_genes = genes[filter_expr_matrix(counts, threshold=threshold)]

    # Write back to the same store the matrix was read from above — writing
    # to `adata.varm` unconditionally (as before) silently discarded this
    # filtering step whenever `use_raw=True`.
    store[database_varm_key][~var_names.isin(valid_genes)] = 0

    return


def filter_genes(
    adata: AnnData,
    layer_key: Literal["use_raw"] | str | None,
    database_varm_key: str,
    cell_type_key: str,
    expression_filt: bool,
    de_filt: bool,
    autocorrelation_filt: bool,
    threshold: float = 0.2,
) -> tuple[list[str], dict[Any, list[str]]]:
    """
    Applies expression and/or DE filtering per cell type.

    Parameters
    ----------
    adata
        Annotated data object (AnnData).
    layer_key
        Which layer to use (or "use_raw").
    database_varm_key
        Key in `adata.varm` pointing to gene database features.
    cell_type_key
        Key in `adata.obs` with cell type labels.
    expression_filt
        Whether to filter based on expression sparsity in each cell type.
    de_filt
        Whether to filter based on differential expression.
    autocorrelation_filt
        Whether to restrict to spatially autocorrelated genes.
    threshold
        Minimum fraction of cells (within each cell type) in which a gene
        must be expressed to pass ``expression_filt``.

    Returns
    -------
    filtered_genes
        List of genes retained across any cell type.
    filtered_genes_ct
        Dict mapping cell types to their filtered genes.
    """
    if autocorrelation_filt:
        autocor_results = adata.uns["gene_autocorrelation_results"]
        sig_genes = autocor_results.query("Z_FDR < 0.05").index
        if len(sig_genes) == 0:
            raise ValueError("There are no significantly autocorrelated genes.")

    else:
        use_raw = layer_key == "use_raw"
        db = adata.raw.varm[database_varm_key] if use_raw else adata.varm[database_varm_key]
        sig_genes = db.loc[(db != 0).any(axis=1)].index

    counts = counts_from_anndata_for_genes(adata, sig_genes, layer_key, dense=True)

    cell_types = adata.obs[cell_type_key].values
    unique_cts = np.unique(cell_types)
    filtered_genes_ct = {}

    # Precompute masks
    masks = {ct: np.where(cell_types == ct)[0] for ct in unique_cts}
    not_masks = {ct: np.where(cell_types != ct)[0] for ct in unique_cts}

    if expression_filt:
        expr_mask = {ct: filter_expr_matrix(counts[:, masks[ct]], threshold) for ct in unique_cts}
    if de_filt:
        de_stats = {
            ct: de_threshold(counts[:, masks[ct]], counts[:, not_masks[ct]]) for ct in unique_cts
        }

    filtered_genes = set()
    for ct in unique_cts:
        gene_mask = np.ones(len(sig_genes), dtype=bool)

        if expression_filt:
            gene_mask &= expr_mask[ct]

        if de_filt:
            stat, pval, cd = de_stats[ct]
            fdr = multipletests(pval, method="fdr_bh")[1]
            gene_mask &= (fdr < 0.05) & (cd > 0)

        selected = sig_genes[gene_mask]
        filtered_genes_ct[ct] = selected.tolist()
        filtered_genes.update(selected)

    return sorted(filtered_genes), filtered_genes_ct


def filter_expr_matrix(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Return genes expressed in at least a threshold fraction of cells."""
    return (matrix > 0).sum(axis=1) / matrix.shape[1] >= threshold


@njit(parallel=True)
def cohens_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Cohen's d row-wise between two matrices."""
    out = np.empty(x.shape[0])

    for i in range(x.shape[0]):
        nx, ny = len(x[i]), len(y[i])
        vx, vy = np.var(x[i], ddof=1), np.var(y[i], ddof=1)
        pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        out[i] = (np.mean(x[i]) - np.mean(y[i])) / pooled if pooled > 0 else 0

    return out


def de_threshold(
    counts_ct: np.ndarray, counts_no_ct: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute differential-expression statistics for cell-type counts."""
    stat = np.array(
        [
            mannwhitneyu(counts_ct[i], counts_no_ct[i], alternative="greater").statistic
            for i in range(counts_ct.shape[0])
        ]
    )
    pval = np.array(
        [
            mannwhitneyu(counts_ct[i], counts_no_ct[i], alternative="greater").pvalue
            for i in range(counts_ct.shape[0])
        ]
    )
    cd = cohens_d(counts_ct, counts_no_ct)

    return stat, pval, cd


def compute_gene_pairs(
    adata: AnnData,
    layer_key: Literal["use_raw"] | str | None = None,
    cell_type_key: str | None = None,
    cell_type_pairs: list | None = None,
    ct_specific: bool | None = True,
    fix_ct: Literal["all"] | str | None = None,
    verbose: bool | None = False,
) -> None:
    """Identify biologically plausible gene pairs.

    This includes ligand-receptor (LR) signaling or metabolite transport pairs based on annotated
    interaction databases and filtered expression data.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (AnnData). Must include:
            - `varm["database"]`: DataFrame indicating gene involvement in interactions.
            - `uns["database"]`: 'LR', 'transporter', or 'both'.
            - `uns["ligand"]`, `uns["receptor"]` for LR pairs if applicable.
            - `uns["metabolite_database"]` and/or `uns["LR_database"]` for pair categorization.
            - `obsp["weights"]`: spatial proximity weights.
    layer_key : str or "use_raw", optional
        Specifies the layer or raw data to use for expression filtering.
    cell_type_key : str, optional
        Key in `adata.obs` indicating cell type annotation.
    cell_type_pairs : list of tuple, optional
        List of tuples specifying cell type pairs to consider. If not provided, all combinations
        are used.
    ct_specific : bool, optional (default: True)
        If True, restrict gene pair computation to combinations relevant to the given cell type
        annotations.
    fix_ct : str, optional
        Whether to restrict the cell type pairs to a particular cell type.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
        Results are stored in the following keys in `adata.uns`: `lcs`, `lc_zs`,
        `lc_z_pvals`, and `lc_z_FDR`.
    """
    start = time.time()
    if verbose:
        print("Computing gene pairs...")

    from_value_to_type = {
        "LR": {-1.0: "REC", 1.0: "LIG"},
        "transporter": {-1.0: "IMP", 1.0: "EXP", 2.0: "IMP-EXP"},
    }

    # Setup
    layer_key = layer_key or adata.uns.get("layer_key")
    use_raw = layer_key == "use_raw"
    genes = adata.raw.var.index if use_raw else adata.var_names
    adata.uns["fix_ct"] = fix_ct

    if ct_specific:
        cell_type_key = cell_type_key or adata.uns.get("cell_type_key")
        if cell_type_key is None:
            raise ValueError('Please provide the "cell_type_key" argument.')
        adata.uns["cell_type_key"] = cell_type_key
        # Cell-type labels live in `adata.obs` regardless of `use_raw`
        # (`Raw` has no `.obs` attribute to read them from).
        cell_types = adata.obs[cell_type_key]
        cell_types = cell_types.values.astype(str)

    # `database` was extracted into `adata.raw.varm` (not `adata.varm`) when
    # `use_raw=True` — see extract_interaction_db — so it must be read from
    # (and, below, written back to) the same store.
    db_store = adata.raw.varm if use_raw else adata.varm
    database = db_store["database"]

    heterodimer_info = adata.uns.get("heterodimer_info")
    if heterodimer_info is not None:
        heterodimer_info = heterodimer_info.copy()
        heterodimer_info["Genes"] = heterodimer_info["Genes"].apply(ast.literal_eval)

    # Filters
    autocorrelation_filt = adata.uns.get("autocorrelation_filt", False)
    expression_filt = adata.uns.get("expression_filt", False)
    de_filt = adata.uns.get("de_filt", False)

    if expression_filt or de_filt:
        filtered_genes = adata.uns["filtered_genes"]
        filtered_genes_ct = adata.uns["filtered_genes_ct"]
    elif autocorrelation_filt:
        autocor_results = adata.uns["gene_autocorrelation_results"]
        filtered_genes = autocor_results[autocor_results.Z_FDR < 0.05].index.tolist()
    else:
        filtered_genes = list(genes)

    if not filtered_genes:
        raise ValueError("No genes have passed the filters.")

    filtered_genes_set = set(filtered_genes)
    all_genes_set = set(genes)
    non_sig_genes = list(all_genes_set - filtered_genes_set)

    # Filter out uninformative metabolites
    database.loc[non_sig_genes] = 0
    cols_keep = [
        col
        for col in database.columns
        if (
            (np.unique(database[col]) != 0).sum() > 1
            or database[col][database[col] != 0].unique().tolist() == [2]
        )
    ]
    database = database[cols_keep].copy()
    db_store["database"] = database

    if ct_specific and "filtered_genes_ct" not in adata.uns:
        filtered_genes_ct = dict.fromkeys(np.unique(cell_types), filtered_genes)
    else:
        filtered_genes_ct = adata.uns.get("filtered_genes_ct", {})

    weights = adata.obsp["weights"]
    if ct_specific:
        if cell_type_pairs is None:
            cell_type_list = list(filtered_genes_ct)
            if fix_ct:
                cell_type_pairs = list(itertools.product(cell_type_list, repeat=2))
                if fix_ct != "all":
                    cell_type_pairs = [pair for pair in cell_type_pairs if pair[0] == fix_ct]
            else:
                cell_type_pairs = list(itertools.combinations_with_replacement(cell_type_list, 2))
        cell_type_pairs_df = pd.Series(cell_type_pairs)
        valid_mask = cell_type_pairs_df.apply(
            get_interacting_cell_type_pairs, args=(weights, cell_types)
        )
        cell_type_pairs = cell_type_pairs_df[valid_mask].tolist()

    # Setup for result aggregation
    gene_pairs_per_metabolite = {}
    gene_pairs = []
    ct_pairs = []
    gene_pairs_per_ct_pair = {} if ct_specific else None

    if adata.uns["database"] == "both":
        metabolites_set = set(adata.uns["metabolite_database"].Metabolite)
        LR_pairs_set = set(adata.uns["LR_database"].index)

    for metabolite in database.columns:
        metab_genes = database.index[database[metabolite] != 0].tolist()
        if not metab_genes:
            continue

        gene_pairs_per_metabolite[metabolite] = {"gene_pair": [], "gene_type": []}

        if adata.uns["database"] == "both":
            if metabolite in metabolites_set:
                int_type = "transporter"
            elif metabolite in LR_pairs_set:
                int_type = "LR"
            else:
                raise ValueError(
                    'The "metabolite" variable needs to be either a metabolite or a LR pair.'
                )
        else:
            int_type = adata.uns["database"]

        # Build gene pairs
        if int_type == "transporter":
            if (
                heterodimer_info is not None
                and metabolite in heterodimer_info["Metabolite"].values
            ):
                for genes_list in heterodimer_info[heterodimer_info["Metabolite"] == metabolite][
                    "Genes"
                ]:
                    if all(g in metab_genes for g in genes_list):
                        metab_genes = [g for g in metab_genes if g not in genes_list] + [
                            tuple(genes_list)
                        ]

            combos = (
                set(itertools.combinations_with_replacement(metab_genes, 2))
                | set(itertools.permutations(metab_genes, 2))
                if ct_specific
                else set(itertools.combinations_with_replacement(metab_genes, 2))
            )
            all_pairs = [
                (list(x) if isinstance(x, tuple) else x, list(y) if isinstance(y, tuple) else y)
                for x, y in combos
            ]

        else:  # LR
            ligand = adata.uns["ligand"].loc[metabolite].dropna().tolist()
            ligand = ligand[0] if len(ligand) == 1 else ligand
            receptor = adata.uns["receptor"].loc[metabolite].dropna().tolist()
            receptor = receptor[0] if len(receptor) == 1 else receptor
            # if len(ligand) == 0 or len(receptor) == 0:
            if not ligand or not receptor:
                continue
            all_pairs = (
                [(ligand, receptor), (receptor, ligand)] if ct_specific else [(ligand, receptor)]
            )

        # Evaluate gene pairs
        for var1, var2 in all_pairs:

            def extract_val(var, metabolite):
                if isinstance(var, str):
                    return database.at[var, metabolite]
                else:
                    vals = list(
                        {
                            database.at[v, metabolite]
                            for v in var
                            if database.at[v, metabolite] != 0
                        }
                    )
                    return vals[0] if len(vals) == 1 else vals

            val1 = extract_val(var1, metabolite)
            val2 = extract_val(var2, metabolite)

            if not val1 or not val2:
                continue
            if val1 == val2 and val1 in (1.0, -1.0):
                continue

            type1 = from_value_to_type[int_type].get(val1)
            type2 = from_value_to_type[int_type].get(val2)

            gene_pairs_per_metabolite[metabolite]["gene_pair"].append((var1, var2))
            gene_pairs_per_metabolite[metabolite]["gene_type"].append((type1, type2))

            if (var1, var2) not in gene_pairs:
                gene_pairs.append((var1, var2))
                if ct_specific:
                    for ct1, ct2 in cell_type_pairs:
                        in_ct1 = (
                            var1 in filtered_genes_ct[ct1]
                            if isinstance(var1, str)
                            else any(v in filtered_genes_ct[ct1] for v in var1)
                        )
                        in_ct2 = (
                            var2 in filtered_genes_ct[ct2]
                            if isinstance(var2, str)
                            else any(v in filtered_genes_ct[ct2] for v in var2)
                        )
                        if in_ct1 and in_ct2:
                            if (ct1, ct2) not in ct_pairs:
                                ct_pairs.append((ct1, ct2))
                            gene_pairs_per_ct_pair.setdefault((ct1, ct2), []).append((var1, var2))

    # Save results
    adata.uns.setdefault("gene_pairs", gene_pairs)
    if ct_specific:
        adata.uns.setdefault("cell_type_pairs", ct_pairs)
        adata.uns.setdefault("gene_pairs_per_ct_pair", gene_pairs_per_ct_pair)
    adata.uns.setdefault("gene_pairs_per_metabolite", gene_pairs_per_metabolite)

    if verbose:
        print("Finished computing gene pairs in %.3f seconds" % (time.time() - start))

    return


def get_interacting_cell_type_pairs(x: tuple, weights: Any, cell_types: np.ndarray) -> bool:
    """Return whether a cell type pair has nonzero spatial weights."""
    ct_1, ct_2 = x

    ct_1_bin = cell_types == ct_1
    ct_2_bin = cell_types == ct_2

    weights = weights.tocsc()
    cell_types_weights = weights[ct_1_bin,][:, ct_2_bin]

    return bool(cell_types_weights.nnz)
