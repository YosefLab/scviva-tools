---
name: scviva
description: Use when running Harreman from scviva-tools (scviva.tools.harreman, HarremanAnalysis) on spatial transcriptomics AnnData to infer spatially resolved cell-cell communication (ligand-receptor pairs, metabolite exchange via transporters), find spatially autocorrelated genes, or group genes into spatial co-expression modules.
---

# scviva-tools: Harreman spatial communication & modules

```python
from scviva.tools import HarremanAnalysis      # stateful wrapper (recommended)
import scviva.tools.harreman as hm             # functional API: hm.pp, hm.tl, hm.hs, hm.pl, hm.ds
```

Every step writes into the AnnData (`adata.uns`, `adata.obsm`) and returns `None`.
`HarremanAnalysis(adata, ...)` works **in place** on the AnnData you pass (no copy).

## Before you start (what guessing gets wrong)

- **Coordinates must be finite.** The KNN graph raises `Input X contains NaN`. Drop those cells first:
  `adata = adata[~np.isnan(np.asarray(adata.obsm[key], float)).any(1)].copy()`.
  Any `obsm` key works as `compute_neighbors_on_key`; it need not be `"spatial"`.
- **Raw counts for count models.** `model="danb"` and `"bernoulli"` expect counts. If counts live in a layer,
  pass `layer_key="counts"` to `HarremanAnalysis(...)`; it becomes the default layer for every step.
- **`species` must match the gene symbols** (`"mouse"` → `Cd74`, `"human"` → `CD74`); `setup` defaults to `"human"`.
  The databases download on first use (needs network).
- `setup(n_neighbors=...)` defaults to **30**. Pass the neighborhood size the task asks for (or `neighborhood_radius=`).
  Pass `use_gpu=False` on CPU machines. If there are several samples, pass `sample_key=` so no edges cross samples.

## Goal: cell-cell communication (LR or metabolite exchange), cell-type agnostic

```python
ha = HarremanAnalysis(adata, layer_key="counts")
ha.setup(compute_neighbors_on_key="spatial", species="mouse",
         database="both",                     # "LR" | "transporter" | "both"
         n_neighbors=K, use_gpu=False)
ha.filter_genes(model="danb", autocorrelation_filt=True)   # optional; the default filters NOTHING
ha.compute_gene_pairs(ct_specific=False)                   # default ct_specific=True needs cell_type_key
ha.compute_cell_communication(mode="standard", model="danb", test="parametric")
ha.select_significant_interactions(fdr_threshold=0.05, test="parametric", use_fdr=True)

sig_gp = adata.uns["ccc_results"]["cell_com_df_gp_sig"]   # same object as ha.results.significant_interactions["gp"]
sig_m  = adata.uns["ccc_results"]["cell_com_df_m_sig"]    # metabolite or LR-interaction level ("m")
ranked = sig_gp.sort_values("Z", ascending=False)         # tables are NOT pre-sorted
```

Defaults that silently change the answer:
- `compute_cell_communication(model=None)` means **no count model** (`"none"`) and gives different Z-scores
  and a different ranking than `"danb"`. Always pass `model=` (match the model used for filtering).
- `test` defaults to `"both"`, which also runs a 1000-permutation test (slow). Use `"parametric"` unless
  permutation results are wanted. `select_significant_interactions` defaults to `test="parametric"`. If you ran
  only `"non-parametric"`, pass `test="non-parametric"`.
- `filter_genes(autocorrelation_filt=True)` keeps genes whose autocorrelation `Z_FDR < 0.05` (a fixed cutoff).
  If none pass, `compute_gene_pairs` raises `No genes have passed the filters.`
- `layer_key_p_test` and `layer_key_np_test` default to the constructor `layer_key`. The tutorial uses raw
  counts for the parametric test and log-normalized counts for the non-parametric one.

Result tables in `adata.uns["ccc_results"]`: `cell_com_df_gp` (all gene pairs) and `cell_com_df_m` (per
metabolite or interaction), plus `*_sig` versions after selection.
- Gene-pair columns: `Gene 1`, `Gene 2`, `C_p`, `Z`, `Z_pval`, `Z_FDR`, `C_norm_p`. The non-parametric test
  adds `C_np`, `pval_np`, `FDR_np`, `C_norm_np`. The `_sig` tables add `selected`.
- Metabolite columns: `Metabolite`, `C_p`, `Z`, `Z_pval`, `Z_FDR` (+ `C_np`, `pval_np`, `FDR_np`).
- `Gene 1` and `Gene 2` can be a **list** for multi-subunit complexes. Join it with `"_"` before comparing strings.
- Metabolite → gene pairs: `adata.uns["gene_pairs_per_metabolite"]`. LR annotation (pathway names): `adata.uns["LR_database"]`.

## Goal: cell-type-resolved communication (which cell types interact)

```python
ha.setup(compute_neighbors_on_key=..., species=..., n_neighbors=K, cell_type_key="<obs column>", use_gpu=False)
ha.filter_genes(model="danb", autocorrelation_filt=True)
ha.compute_gene_pairs()                                   # ct_specific=True uses cell_type_key
ha.compute_cell_communication(mode="cell_type", model="danb", test="parametric")
ha.select_significant_interactions()                      # ct_aware is automatically True after mode="cell_type"
df = adata.uns["ct_ccc_results"]["cell_com_df_gp_sig"]    # adds "Cell Type 1", "Cell Type 2" columns
```
The results go to `ct_ccc_results`, not to `ccc_results`.

## Goal: per-spot interaction scores (where an interaction is active)

After selection: `ha.compute_interacting_cell_scores(mode=<same as CCC>, test="parametric")`.
In standard mode the scores are at `adata.uns["interacting_cell_results"][t][kind]["cs"]`, where `t` is
`"p"` (parametric) or `"np"` (non-parametric) and `kind` is `"gp"` or `"m"`. These are arrays with one row
per `adata.obs_names`. The columns are `adata.uns["gene_pairs_sig_names"]` (gp, e.g. `"Lig_Rec"`) and
`adata.uns["metabolites"]` (m). Each entry also has `Z`, `Z_pval`, `Z_FDR`.
In cell-type mode the scores are at `adata.uns["ct_interacting_cell_results"]`.

## Goal: spatially variable genes, gene modules, zonation

```python
ha = HarremanAnalysis(adata)
ha.tl.compute_knn_graph(compute_neighbors_on_key="spatial", n_neighbors=K, use_gpu=False)  # no database needed
ha.hs.compute_local_autocorrelation(layer_key="counts", model="danb")      # always pass model
ac = adata.uns["gene_autocorrelation_results"]       # index = gene; columns C, Z, Z_Pval, Z_FDR
genes = ac[ac.Z_FDR < 0.05].sort_values("Z", ascending=False).index
ha.hs.compute_local_correlation(genes=genes)         # -> uns: lcs, lc_zs, lc_z_pvals, lc_z_FDR
ha.hs.create_modules(min_gene_threshold=20, fdr_threshold=0.05)  # -> uns["modules"]: Series gene -> module id (-1 = none)
ha.hs.calculate_module_scores()                      # -> obsm["module_scores"], columns "Module 1", ...
labels = ha.hs.compute_top_scoring_modules(sd=1)     # returns per-cell labels; assign to obs yourself
```
- `use_metabolic_genes=True, species=...` restricts autocorrelation to metabolic enzymes. The `species`
  default here is `"mouse"`, unlike `setup`.
- Genes per module: `adata.uns["gene_modules"]` (dict `"Module k"` → genes). Super-modules:
  `ha.hs.calculate_super_module_scores(super_module_dict={new_id: [module ids]})` → `obsm["super_module_scores"]`.
  Then pass `use_super_modules=True` to later calls.
- To link communication to modules: `ha.tl.compute_interaction_module_correlation(cor_method="pearson", interaction_type="metabolite", test=...)`.
- For pathway enrichment of modules (VISION), plots, or feeding DestVI, ResolVI or SCVIVA outputs in, see `references/harreman_extras.md`.

## Functional API (same outputs)

`ha.setup` = `hm.pp.extract_interaction_db(adata, species=, database=)` + `hm.tl.compute_knn_graph(adata, compute_neighbors_on_key=, n_neighbors=)`.
The remaining steps are:
- `hm.tl.apply_gene_filtering(adata, layer_key=, model=, autocorrelation_filt=)`
- `hm.tl.compute_gene_pairs(adata, ct_specific=False)`
- `hm.tl.compute_cell_communication(adata, layer_key_p_test=, model=, test=)`. Here the number of permutations is `M=`, not `n_permutations`.
- `hm.tl.select_significant_interactions(adata, test=, use_FDR=, threshold=)`

For cell types, use `hm.tl.compute_ct_cell_communication(..., cell_type_key=)` and then `select_significant_interactions(adata, ct_aware=True)`.

`HarremanAnalysis` enforces the step order (`setup` → `compute_gene_pairs` → `compute_cell_communication` →
selection and scores) and raises `RuntimeError` if you skip a step. Re-running `setup` invalidates the steps after it.
