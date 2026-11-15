# Imputation and multi-modal integration: GIMVI, DIAGVI

## GIMVI: impute genes the spatial panel did not measure

```python
import scanpy as sc
from scviva.model import GIMVI

# spatial genes must be a subset of the sequencing genes
sc.pp.filter_cells(spatial, min_counts=1); sc.pp.filter_cells(seq, min_counts=1)
GIMVI.setup_anndata(seq, layer=None, labels_key=None, batch_key=None)       # set what you have
GIMVI.setup_anndata(spatial, layer=None, labels_key=None, batch_key=None)
model = GIMVI(seq, spatial)          # order matters: (adata_seq, adata_spatial)
model.train(max_epochs=...)          # default 200
imputed_seq, imputed_spatial = model.get_imputed_values(normalized=True)
# imputed_spatial: ndarray (n_spatial_cells x n_seq_genes), columns in seq.var_names order
latent_seq, latent_spatial = model.get_latent_representation()
```

- Both methods return a **list** `[seq_result, spatial_result]`, not a DataFrame. Wrap:
  `pd.DataFrame(imputed_spatial, index=spatial.obs_names, columns=seq.var_names)`.
- To evaluate imputation of a gene, hold it out of `spatial` before training, then compare
  the imputed column with the measured values (e.g. Spearman correlation).
- Spatial data has to be counts; cells with zero counts break training (filter first).

## DIAGVI: unpaired modalities (RNA + spatial proteomics or spatial RNA)

```python
from scviva.external import DIAGVI
DIAGVI.setup_anndata(adata_rna, layer="counts", labels_key="cell_type", likelihood="nb")
DIAGVI.setup_anndata(adata_prot, likelihood="normal")   # transformed, scaled protein data
model = DIAGVI({"rna": adata_rna, "protein": adata_prot})   # dict keys = modality names
# feature links: default = shared feature names; else mapping_df (one column per modality name,
# one row per linked feature pair) or a prebuilt guidance_graph
model.train(max_epochs=...)
lat = model.get_latent_representation()      # dict modality -> ndarray
imp = model.get_imputed_values(query_name="rna", query_adata=adata_rna)  # other modality's features for rna cells
preds = model.predict_celltype(labeled_modality="rna")   # dict with "predictions", "confidence"
```

- Likelihood per modality: `"nb"` for raw counts, `"normal"` for transformed and scaled
  continuous data (others: zinb, nbmixture, log1pnormal, ziln, zig).
- `predict_celltype` needs `labels_key` set on the labeled modality at setup.
- Alignment strength: pass `model.train(plan_kwargs={"lam_sinkhorn": ..., "lam_class": ...})`
  (defaults 85 and 5, which favor strong alignment; lower `lam_sinkhorn` and raise
  `lam_class` for similar modalities).
