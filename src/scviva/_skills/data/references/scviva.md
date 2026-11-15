# SCVIVA: niche-aware model and niche DE

Needs single-cell-resolution data (not Visium spots) with: raw counts, coordinates in
`obsm`, a cell-type column, a sample column, and an expression embedding in `obsm`
(scVI, scANVI, PCA... any per-cell embedding).

```python
import scviva
from scviva.model import SCVIVA

scviva.settings.seed = 0
keys = dict(
    sample_key="sample",                   # obs column; REQUIRED (None -> KeyError)
    labels_key="cell_type",                # obs column with cell types
    cell_coordinates_key="spatial",        # obsm key with x,y
    expression_embedding_key="X_scVI",     # obsm key; if absent, compute one (e.g. sc.pp.pca -> "X_pca")
)
# 1. niche preprocessing: writes obsm niche_indexes, niche_distances, niche_composition, niche_activation
SCVIVA.preprocessing_anndata(adata, k_nn=20, **keys)
# 2. setup with the SAME keys again (setup does not remember step 1's arguments)
SCVIVA.setup_anndata(adata, layer="counts", batch_key="sample", **keys)
model = SCVIVA(adata)
model.train()   # pass max_epochs, early_stopping, batch_size, plan_kwargs={"lr": ...} if given
```

- No sample column? Create one: `adata.obs["sample"] = "s0"`. Neighbors are computed within
  each sample.
- If the embedding has to be computed, do it before step 1; step 1 averages it per cell type
  over each cell's `k_nn` neighbors.

## Outputs

- `model.get_latent_representation()` → ndarray (n_cells × n_latent); store in `obsm` and use
  `sc.pp.neighbors(adata, use_rep=...)` for clustering.
- `model.predict_neighborhood()` → ndarray (n_cells × n_cell_types), predicted niche composition.
- `model.predict_niche_activation()` → ndarray, predicted neighbor embedding per cell type.
- `model.get_normalized_expression()` → DataFrame cells × genes.

## Niche-aware differential expression

```python
res = model.differential_expression(
    adata, groupby="cell_type", group1="TypeA", group2="TypeB",  # group2=None -> vs rest
    k_nn=..., delta=..., fdr_target=...,   # pass task values; defaults delta=0.15, fdr_target=0.05
)
df = res.g1_g2                             # group1 vs group2 table, indexed by gene
top = df.sort_values("lfc_mean", ascending=False).index[0]
```

- `niche_mode=True` is the default and returns a `DifferentialExpressionResults` dataclass,
  **not** a DataFrame: `.g1_g2` (group1 vs group2), `.g1_n1` (group1 vs its spatial
  neighbors of other types), `.n1_g2`, `.n1_n2`, `.gpc` (classifier). Every table has
  `lfc_mean`, `proba_de`, `is_de_fdr_<fdr>`, and `proba_de_g1_n1` = probability the gene is a
  true local marker of group1 rather than neighbor contamination.
- For "upregulated in group1, not explained by neighbors": rank `g1_g2` by `lfc_mean`, and
  if asked to filter contamination keep genes with high `proba_de_g1_n1` (e.g. > 0.5).
- `niche_mode=False` runs plain scVI DE and returns a single DataFrame.
- The neighbor graph for niche DE is cached in `adata.uns["adjusted_A"]` on the first call;
  later calls with a different `k_nn` or `radius` silently reuse it. Run
  `del adata.uns["adjusted_A"]` before re-running with new neighborhood settings.
- `ValueError: No DE genes found between group1 and neighbors1` means no local markers passed
  the FDR (tiny groups, undertrained model). Train longer, or relax `fdr_target` or `delta`
  if the task allows; otherwise report it.
- `model.differential_abundance(...)` exists for sample-level abundance comparisons.
