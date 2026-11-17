# scVIVA: niche-aware embedding and niche DE

Needs raw counts (layer), a cell-type column, spatial coordinates in `obsm`, and an expression
embedding in `obsm`. If no embedding exists, train `scvi.model.SCVI` (or SCANVI) and store
its latent representation in `obsm` first.

```python
import scvi, scviva
from scviva.model import SCVIVA
scvi.settings.seed = 0
kw = dict(sample_key="<slide_col>", labels_key="<celltype_col>",
          cell_coordinates_key="spatial", expression_embedding_key="X_scVI")  # use the keys that exist
SCVIVA.preprocessing_anndata(adata, k_nn=20, **kw)          # must run BEFORE setup_anndata
SCVIVA.setup_anndata(adata, layer="counts", batch_key="<slide_col>", **kw)
model = SCVIVA(adata)
model.train(max_epochs=..., early_stopping=True, batch_size=512, plan_kwargs={"lr": 5e-4})  # tutorial settings
adata.obsm["X_scVIVA"] = model.get_latent_representation()

res = model.differential_expression(adata, groupby="<col>", group1=["A"], group2="B",
                                    niche_mode=True, k_nn=..., delta=..., fdr_target=...)
df = res.g1_g2          # main table: genes x stats, includes lfc_mean, proba_de, is_de_fdr_*
```

- `res` is a `DifferentialExpressionResults` dataclass. It also has `g1_n1` (group 1 vs its
  neighbourhood), `n1_g2`, `n1_n2`, `gpc` and `.plot()`.
- `niche_mode=False` gives plain scVI-style DE. It returns a DataFrame.
- `delta` may be a list of four values, one per comparison. Extra kwargs such as
  `pseudocounts` and `n_samples_overall` go to `get_bayes_factors`.
- To rank genes up in group 1, sort `df` by `lfc_mean` (descending), after filtering on the
  DE flag or `proba_de` if the task asks for significance.
