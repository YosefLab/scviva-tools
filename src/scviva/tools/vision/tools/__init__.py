from .knn import (
    compute_knn_weights,
    compute_knn_weights_anndata,
    compute_knn_weights_from_tree,
    compute_knn_weights_from_tree_anndata,
    compute_knn_weights_from_tree_lca,
    compute_knn_weights_from_tree_lca_anndata,
    find_knn,
)
from .microclusters import (
    apply_micro_clustering,
    pool_matrix,
    pool_matrix_anndata,
    pool_metadata,
    pool_metadata_anndata,
)
from .projections import (
    apply_ica,
    apply_isomap,
    apply_pca,
    apply_permutation_wpca,
    apply_rbfpca,
    apply_tsne,
    apply_umap,
    compute_latent_space,
    generate_projections,
    log2p1,
)
from .signature import (
    compute_obs_df_scores,
    compute_signature_scores,
    compute_signatures_anndata,
    load_signatures,
    split_signed_signatures,
)

__all__ = [
    "apply_ica",
    "apply_isomap",
    "apply_micro_clustering",
    "apply_pca",
    "apply_permutation_wpca",
    "apply_rbfpca",
    "apply_tsne",
    "apply_umap",
    "compute_knn_weights",
    "compute_knn_weights_anndata",
    "compute_knn_weights_from_tree",
    "compute_knn_weights_from_tree_anndata",
    "compute_knn_weights_from_tree_lca",
    "compute_knn_weights_from_tree_lca_anndata",
    "compute_latent_space",
    "compute_obs_df_scores",
    "compute_signature_scores",
    "compute_signatures_anndata",
    "find_knn",
    "generate_projections",
    "load_signatures",
    "log2p1",
    "pool_matrix",
    "pool_matrix_anndata",
    "pool_metadata",
    "pool_metadata_anndata",
    "split_signed_signatures",
]

# NOTE: `.diffexp` (rank_genes_groups) is intentionally NOT imported here --
# it imports scanpy at module level, and importing it eagerly from this
# package's __init__ would pull in all of scanpy just by `import scviva`.
# It is exposed lazily via `scviva.tools.vision.__getattr__`.
