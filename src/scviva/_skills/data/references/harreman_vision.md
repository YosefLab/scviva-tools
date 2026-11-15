# Harreman (cell-cell and metabolic communication) and VISION (signature scoring)

Both are stateful session objects. Steps must run in order; a skipped prerequisite raises
`RuntimeError: <step>() requires <prereq>() to be run first`.

## HarremanAnalysis

```python
import numpy as np
from scviva.tools import HarremanAnalysis

coords = np.asarray(adata.obsm["spatial"], dtype=float)
adata = adata[np.isfinite(coords).all(axis=1)].copy()      # drop cells with NaN coordinates

ha = HarremanAnalysis(adata, layer_key="counts")           # layer with raw counts; None -> X
ha.setup(compute_neighbors_on_key="spatial",               # obsm key used for the kNN graph
         species="mouse",                                  # default "human" -- set it!
         database="both",                                  # "LR", "transporter", or "both"
         n_neighbors=30,                                   # default; or neighborhood_radius=
         cell_type_key=None)                               # obs column; needed for ct-specific steps
ha.filter_genes(model="danb", autocorrelation_filt=True)  # optional: keep spatially autocorrelated genes
ha.compute_gene_pairs(ct_specific=False)                   # default True REQUIRES cell_type_key
ha.compute_cell_communication(mode="standard", model="danb", test="parametric")  # test default "both"
ha.select_significant_interactions(fdr_threshold=0.05, test="parametric", use_fdr=True)

ccc = ha.adata.uns["ccc_results"]          # "ct_ccc_results" when mode="cell_type"
gp_sig = ccc["cell_com_df_gp_sig"]          # significant gene pairs: "Gene 1", "Gene 2", "Z", "Z_FDR", "C_p", ...
m_sig = ccc["cell_com_df_m_sig"]            # significant metabolites or LR pathways
top = gp_sig.sort_values("Z", ascending=False).iloc[0]
```

- Results are written at the **top level** of `adata.uns` (`gene_autocorrelation_results`,
  `gene_pairs`, `ccc_results`, `ct_ccc_results`, `interacting_cell_results`), not under
  `adata.uns["harreman"]` (that holds only params). `ha.results` gives the same objects:
  `ha.results.significant_interactions` is `{"gp": ..., "m": ...}`.
- Unfiltered tables: `ccc["cell_com_df_gp"]` and `ccc["cell_com_df_m"]`. Parametric
  statistics carry a `_p` suffix (plus `Z`, `Z_pval`, `Z_FDR`); non-parametric ones carry
  `_np`. `select_significant_interactions(test=...)` must name a test that was computed.
- `compute_interacting_cell_scores(...)` (after communication) gives per-cell scores.
- Cell-type-resolved communication: set `cell_type_key` in `setup`, then
  `compute_gene_pairs(ct_specific=True)` and `compute_cell_communication(mode="cell_type")`.
- Model integration: `HarremanAnalysis(adata, model=trained_model)` accepts DestVI, ResolVI or
  SCVIVA; with DestVI it adds one proportion-scaled count layer per cell type.
- Lower-level functional API: `scviva.tl.harreman.tl.compute_knn_graph`,
  `.hs.compute_local_autocorrelation`, `.hs.compute_local_correlation`,
  `.tl.compute_cell_communication`. Session accessors `ha.hs`, `ha.tl`, `ha.pl`, `ha.vs`
  forward to these with `adata` filled in.

## VisionAnalysis

```python
from scviva.tools import VisionAnalysis
va = VisionAnalysis(adata, norm_data_key=None)   # layer with normalized, NOT log-transformed data
va.setup(compute_neighbors_on_key="X_pca")      # obsm latent; or va.compute_latent_space() first, then va.setup()
va.load_signatures(gmt_files=["sets.gmt"])      # or dicts=[{"SigName": ["GENE1", "GENE2", ...]}]
# signed sets: names ending _up and _down (e.g. "Sig_up", "Sig_down") merge into one signature;
# signatures with < 5 matching genes are dropped (min_signature_genes)
va.compute_signatures()                         # per-cell scores + Geary's C autocorrelation
va.compute_differential_expression()            # one-vs-all over categorical obs columns
r = va.results
r.signature_scores                # DataFrame cells x signatures (obsm["vision_signatures"])
r.signature_autocorrelation       # DataFrame: c_prime (higher = more coherent on the graph), pvals, fdr
r.signature_differential          # dict obs column -> DataFrame
```

- On a Harreman session the same analysis reuses the spatial graph:
  `ha.vs.load_signatures(...)`, then `ha.vs.analyze_vision(signature_varm_key="signatures")`.
