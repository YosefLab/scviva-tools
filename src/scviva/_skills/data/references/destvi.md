# DestVI: per-spot cell-type proportions from a scRNA-seq reference

```python
import numpy as np, scanpy as sc, scvi, scviva
scvi.settings.seed = 0
# 1. Raw counts in a layer for both objects. Choose genes on the reference (e.g. HVGs with
#    flavor="seurat_v3" on counts), then subset both objects to the shared genes.
sc_ad.layers["counts"] = sc_ad.X.copy(); st.layers["counts"] = st.X.copy()
genes = np.intersect1d(sc_ad.var_names, st.var_names)
sc_ad = sc_ad[:, genes].copy(); st = st[:, genes].copy()
# 2. Reference model. labels_key = the cell-type column whose proportions you want.
scvi.model.CondSCVI.setup_anndata(sc_ad, layer="counts", labels_key="<celltype_col>")  # batch_key optional
sc_model = scvi.model.CondSCVI(sc_ad, weight_obs=False)
sc_model.train(max_epochs=...)
# 3. Spatial model. Use scviva.DestVI, NOT scvi.model.DestVI. from_rna_model sets up st itself.
st_model = scviva.DestVI.from_rna_model(st, sc_model)
st_model.train(max_epochs=...)          # the default is 2000, so set it explicitly
props = st_model.get_proportions()      # DataFrame: spots x cell types (rows sum to 1)
```

- If CondSCVI was set up with `batch_key="b"`, then `st.obs["b"]` must exist. Otherwise set
  up CondSCVI without a batch key.
- Compare conditions by grouping spots on an `st.obs` column, e.g.
  `props[mask_a].mean() - props[mask_b].mean()`. Use `props.idxmax(axis=1)` to get each spot's
  dominant cell type.
- Other outputs: `get_gamma()` gives the per-cell-type latent states.
  `get_scale_for_ct(ct)` and `get_expression_for_ct(ct)` give cell-type-specific expression
  in each spot.
- Do not monkeypatch `get_vamp_prior` or force CPU. Those patches are only needed for
  `scvi.model.DestVI`, which you should not use.
