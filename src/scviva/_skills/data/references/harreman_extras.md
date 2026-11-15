# Harreman extras: VISION pathway enrichment, plotting, model integration

## Pathway enrichment of spatial modules (VISION)

This step needs modules and module scores first (`create_modules` + `calculate_module_scores`, or super-modules).
```python
ha.vs.signatures_from_file(dicts=[{"PathwayA": [genes...], ...}])   # or gmt_files=[path]; -> adata.varm["signatures"]
ha.vs.analyze_vision(norm_data_key="log_norm", scores_only=True)    # layer with LOG-NORMALIZED data (None = adata.X)
ha.vs.integrate_vision_hotspot_results(use_super_modules=False)     # cor_method="pearson" default
stats = adata.uns["sig_mod_enrichment_stats"]   # pathways x modules; also "sig_mod_enrichment_pvals", "sig_mod_enrichment_FDR"
```
`load_signatures` drops signatures with fewer than `min_signature_genes=5` genes present in the data.

## Plotting (`ha.pl.*`, or `hm.pl.*(adata, ...)`)

- `local_correlation_plot(use_super_modules=False)`, `average_local_correlation_plot()`, `module_score_correlation_plot()`
- `plot_interacting_cell_scores(interactions=[...], test="parametric", coords_obsm_key="spatial")`. The
  interactions are metabolite names or gene-pair names from `uns["gene_pairs_sig_names"]`.
- `plot_interaction_module_correlation(threshold=...)` (after `ha.tl.compute_interaction_module_correlation`)
- Cell-type variants: `plot_ct_interacting_cell_scores`, `plot_ct_interaction`.
Pass `show=False` where it is available, and save with `matplotlib.pyplot.savefig` in headless runs.

## Feeding trained scviva models into Harreman

`HarremanAnalysis(adata, model=trained_model)` accepts DestVI, ResolVI or SCVIVA (from `scviva`):
- DestVI: for each cell type it adds `adata.layers[<cell type>] = X * proportion` (it uses `adata.X`,
  so X must hold counts). Then `setup(..., cell_type_key=...)` and run `compute_cell_communication(mode="cell_type")`.
- ResolVI: the denoised expression goes to `adata.layers["harreman_denoised"]`. Pass that as `layer_key`
  or `layer_key_p_test` to use it.
- SCVIVA: the latent goes to `adata.obsm["X_harreman_latent"]`. Use it as `compute_neighbors_on_key` for a latent-space graph.

## Bundled example data

`hm.ds.load_visium_mouse_colon_dataset(sample=None, save_path="data")` and `hm.ds.load_slide_seq_human_lung_dataset(...)`
download example AnnData. Prefer the file the task gives you.
