# ResolVI: denoising, semi-supervised relabeling, DE on cellular-resolution ST

`scviva.model.ResolVI` (same model as `scvi.external.RESOLVI`; prefer the scviva class).

```python
import scviva
from scviva.model import ResolVI

scviva.settings.seed = 0
ResolVI.setup_anndata(
    adata, layer="counts",
    labels_key="cell_type",                       # only needed for semisupervised=True
    prepare_data_kwargs={"spatial_rep": "spatial"},  # obsm key with coordinates
)
model = ResolVI(adata, semisupervised=True)       # False for unsupervised denoising only
model.train(max_epochs=...)                       # default 50
```

- Coordinates: by default setup looks for `obsm["X_spatial"]` and raises
  `KeyError: Spatial key 'X_spatial' not found` otherwise. Either pass
  `prepare_data_kwargs={"spatial_rep": "<your obsm key>"}` or copy the coordinates to
  `obsm["X_spatial"]`. Setup builds `obsm["index_neighbor"]` and `obsm["distance_neighbor"]`
  (`n_neighbors=10` by default, also set through `prepare_data_kwargs`).
- Every cell needs > 0 counts and `obs_names` must be unique (asserted in setup).
- Unlabeled cells: label value `"unknown"` (`unlabeled_category`).

## Outputs

- `model.predict(adata, soft=False, num_samples=...)` → ndarray of label strings (hard
  labels). Default `num_samples=30`; pass the task's value. `soft=True` → DataFrame cells x
  labels of probabilities. Store hard labels in a new obs column for downstream grouping.
- `model.get_latent_representation()` → ndarray (n_cells × n_latent).
- `model.get_normalized_expression(...)` → denoised expression DataFrame.
- `model.get_neighbor_abundance(...)`, `model.differential_niche_abundance(groupby=..., ...)`
  for neighborhood composition.

## Differential expression

```python
de = model.differential_expression(
    adata, groupby="new_labels", group1="TypeA", group2="TypeB",
    mode="change", delta=..., weights="importance", filter_outlier_cells=True,
    pseudocounts=..., test_mode="three",   # pass through only what the task specifies
)
top = de.sort_values("lfc_mean", ascending=False).index[0]
```

Returns a DataFrame indexed by gene (default `delta=0.25`, `weights="uniform"`); extra kwargs
like `pseudocounts`, `test_mode` are forwarded to the Bayes-factor computation.
