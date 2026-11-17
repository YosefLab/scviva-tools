# ResolVI: denoising, label correction (semisupervised), DE

```python
import scvi, scviva, pandas as pd
scvi.settings.seed = 0
# Neighbours are computed from obsm["X_spatial"] by default. Point to your coordinate key:
scviva.ResolVI.setup_anndata(adata, layer="counts", labels_key="<label_col>",
                             prepare_data_kwargs={"spatial_rep": "spatial"})
model = scviva.ResolVI(adata, semisupervised=True)   # semisupervised needs labels_key; "unknown" = unlabeled
model.train(max_epochs=...)                          # default 50
adata.obs["resolvi_label"] = pd.Categorical(model.predict(adata, soft=False, num_samples=...))
# predict(soft=False) returns a numpy array of label strings; soft=True returns a DataFrame of class probabilities
de = model.differential_expression(adata, groupby="resolvi_label", group1=["A"], group2="B",
                                   mode="change", delta=..., weights="importance",
                                   filter_outlier_cells=True)   # returns a DataFrame (genes x stats)
```

- `scviva.ResolVI`, `scviva.model.ResolVI` and `scvi.external.RESOLVI` all work. Use the
  scviva one.
- Extra DE kwargs such as `pseudocounts` and `test_mode` are forwarded to
  `get_bayes_factors`.
- Denoised expression: `model.get_normalized_expression()`. Latent space:
  `model.get_latent_representation()`.
