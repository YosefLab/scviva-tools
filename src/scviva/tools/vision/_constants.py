# src/scviva/tools/vision/_constants.py
from __future__ import annotations

# ── adata.uns namespace for run config (see VisionAnalysis.setup) ────────────
VISION_UNS_KEY = "vision"
VISION_PARAMS_KEY = "params"

# ── Step names (used in _completed_steps set) ─────────────────────────────────
STEP_SETUP = "setup"
STEP_LATENT = "compute_latent_space"
STEP_SIGNATURES = "compute_signatures"
STEP_DE = "compute_differential_expression"

# ── Graph / clustering keys written by setup() ────────────────────────────────
WEIGHTS_OBSP_KEY = "weights"
CLUSTERS_OBS_KEY = "VISION_Clusters"
TREE_CLUSTERS_OBS_KEY = "VISION_Clusters_Tree"
TREE_UNS_KEY = "vision_tree"

# ── Signature keys ─────────────────────────────────────────────────────────────
SIGNATURES_VARM_KEY = "signatures"
SIGNATURES_OBSM_KEY = "vision_signatures"
NORM_DATA_KEY_UNS_KEY = "norm_data_key"
SIGNATURE_VARM_KEY_UNS_KEY = "signature_varm_key"

# ── Analysis result keys written by compute_differential_expression() ────────
OBS_DF_SCORES_UNS_KEY = "vision_obs_df_scores"
SIGNATURE_SCORES_UNS_KEY = "vision_signature_scores"
SIGNATURE_DIFFERENTIAL_UNS_KEY = "vision_signature_differential"
META_DIFFERENTIAL_UNS_KEY = "vision_meta_differential"
GENE_IMPORTANCE_UNS_KEY = "vision_gene_importance"
DENDROGRAM_UNS_KEY = "vision_dendrogram"
SIG_CLUSTERS_UNS_KEY = "vision_sig_clusters"
LCA_UNS_KEY = "vision_lca"
LCA_META_UNS_KEY = "vision_lca_meta"

# ── Protein (CITE-seq) analysis keys ──────────────────────────────────────────
PROTEIN_AUTOCORR_UNS_KEY = "vision_protein_autocorr"
PROTEIN_DIFFERENTIAL_UNS_KEY = "vision_protein_differential"
