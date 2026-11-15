# Deconvolution and mapping: DestVI, Stereoscope, Tangram

All three need an scRNA reference with cell-type labels and a spatial AnnData. **Restrict
both to the shared genes first** (same genes, same order):

```python
import numpy as np, scanpy as sc
shared = np.intersect1d(sc_adata.var_names, st_adata.var_names)
sc_adata, st_adata = sc_adata[:, shared].copy(), st_adata[:, shared].copy()
```

Typical reference preprocessing (from the package tutorials): `sc.pp.filter_genes(sc_adata,
min_counts=10)`, keep raw counts in `layers["counts"]`, then
`sc.pp.highly_variable_genes(sc_adata, n_top_genes=2000, flavor="seurat_v3", layer="counts",
subset=True)` before intersecting. Drop near-empty spots (e.g. total counts <= 10). Follow the
task's numbers if it gives any.

## DestVI (two stages)

```python
from scvi.model import CondSCVI
from scviva.model import DestVI

CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="cell_type")  # batch_key optional
sc_model = CondSCVI(sc_adata, weight_obs=False)
sc_model.train(max_epochs=...)          # scvi default is 300; tutorials use 100

st_model = DestVI.from_rna_model(st_adata, sc_model)   # registers st_adata itself
st_model.train(max_epochs=...)          # default 2000; tutorials use 250
props = st_model.get_proportions()      # DataFrame spots x cell types, rows sum to 1
```

- `from_rna_model` re-runs `DestVI.setup_anndata` on `st_adata` with the reference's setup
  arguments, so `st_adata` must have the **same layer name** as used for CondSCVI (e.g. put
  raw spot counts in `st_adata.layers["counts"]`). Extra DestVI setup args go in
  `anndata_setup_kwargs={...}` (e.g. `{"smoothed_layer": "smoothed"}`).
- CondSCVI's default `prior="normal"` works with DestVI in 0.1.8; no monkeypatching needed.
- `get_proportions(keep_additional=False)` drops the extra `add_celltypes` columns.
- Cell-type-specific expression per spot: `st_model.get_scale_for_ct("TypeName")` and
  `get_expression_for_ct(...)` → DataFrame spots x genes. Cell-type-specific latent states:
  `st_model.get_gamma()`.
- Comparing conditions: average `props` over spots in each condition (match by
  `st_adata.obs` index) and take the difference; argmax gives the most-enriched type.

## Stereoscope

```python
from scviva.external import RNAStereoscope, SpatialStereoscope
RNAStereoscope.setup_anndata(sc_adata, layer="counts", labels_key="cell_type")
sc_model = RNAStereoscope(sc_adata); sc_model.train(max_epochs=...)   # default 400
SpatialStereoscope.setup_anndata(st_adata, layer="counts")
sp_model = SpatialStereoscope.from_rna_model(st_adata, sc_model); sp_model.train(max_epochs=...)
props = sp_model.get_proportions()     # DataFrame spots x cell types
```

## Tangram (MuData, `setup_mudata`, no `setup_anndata`)

```python
import mudata
from scviva.external import Tangram
mdata = mudata.MuData({"sp": st_adata, "sc": sc_adata,
                       "sp_train": st_adata[:, genes].copy(), "sc_train": sc_adata[:, genes].copy()})
st_adata.obs["density_prior"] = ...    # optional; must sum to 1
Tangram.setup_mudata(mdata, density_prior_key="density_prior",   # or None
                     modalities={"density_prior_key": "sp_train",
                                 "sc_layer": "sc_train", "sp_layer": "sp_train"})
model = Tangram(mdata)                 # constrained=True requires target_count=<n cells>
model.train()                          # default 1000 epochs, lr=0.1
mapper = model.get_mapper_matrix()     # ndarray (n_sc_cells x n_spots), rows sum to 1
ct = model.project_cell_annotations(mdata["sc"], mdata["sp"], mapper, mdata["sc"].obs["cell_type"])
# ct: DataFrame spots x label categories
proj = model.project_genes(mdata["sc"], mdata["sp"], mapper)   # AnnData, spots x sc genes
```

- `genes` = shared (marker) genes; the two training modalities must have identical genes in
  identical order. Make `var_names` unique in each modality and call `mdata.update()`.
