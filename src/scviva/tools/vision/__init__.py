from ._analysis import VisionAnalysis, VisionToolsAccessor
from ._results import VisionResults
from .phylo import cluster_cells_tree, compute_plasticity_scores
from .preprocessing import (
    NormData,
    apply_filters,
    filter_genes_fano,
    filter_genes_novar,
    filter_genes_threshold,
    get_normalized_copy,
    get_normalized_copy_sparse,
)
from .tools import (
    apply_ica,
    apply_isomap,
    apply_micro_clustering,
    apply_pca,
    apply_permutation_wpca,
    apply_rbfpca,
    apply_tsne,
    apply_umap,
    compute_knn_weights,
    compute_knn_weights_anndata,
    compute_knn_weights_from_tree,
    compute_knn_weights_from_tree_anndata,
    compute_knn_weights_from_tree_lca,
    compute_knn_weights_from_tree_lca_anndata,
    compute_latent_space,
    compute_obs_df_scores,
    compute_signature_scores,
    compute_signatures_anndata,
    find_knn,
    generate_projections,
    load_signatures,
    log2p1,
    pool_matrix,
    pool_matrix_anndata,
    pool_metadata,
    pool_metadata_anndata,
    split_signed_signatures,
)

__all__ = [
    "NormData",
    "VisionAnalysis",
    "VisionResults",
    "VisionToolsAccessor",
    "apply_filters",
    "apply_ica",
    "apply_isomap",
    "apply_micro_clustering",
    "apply_pca",
    "apply_permutation_wpca",
    "apply_rbfpca",
    "apply_tsne",
    "apply_umap",
    "cluster_cells_tree",
    "compute_knn_weights",
    "compute_knn_weights_anndata",
    "compute_knn_weights_from_tree",
    "compute_knn_weights_from_tree_anndata",
    "compute_knn_weights_from_tree_lca",
    "compute_knn_weights_from_tree_lca_anndata",
    "compute_latent_space",
    "compute_obs_df_scores",
    "compute_plasticity_scores",
    "compute_signature_scores",
    "compute_signatures_anndata",
    "filter_genes_fano",
    "filter_genes_novar",
    "filter_genes_threshold",
    "find_knn",
    "generate_projections",
    "get_normalized_copy",
    "get_normalized_copy_sparse",
    "load_signatures",
    "log2p1",
    "pool_matrix",
    "pool_matrix_anndata",
    "pool_metadata",
    "pool_metadata_anndata",
    "rank_genes_groups",
    "split_signed_signatures",
]


def __getattr__(name: str):
    # `.tools.diffexp` imports scanpy at module level, so it's kept out of the
    # eager imports above -- otherwise plain `import scviva` would always
    # pull in all of scanpy (and trigger its numba cache setup) just to
    # reach `HarremanAnalysis`/`VisionAnalysis`.
    if name == "rank_genes_groups":
        from .tools.diffexp import rank_genes_groups

        return rank_genes_groups
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
