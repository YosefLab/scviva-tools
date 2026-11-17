# Harreman: spatial cell-cell and metabolic communication

```python
import numpy as np, scanpy as sc
from scviva.tools.harreman import HarremanAnalysis
coords = "spatial"                                       # the obsm key holding coordinates
adata = adata[~np.isnan(adata.obsm[coords]).any(axis=1)].copy()   # NaN coordinates break the KNN step
adata.obsm[coords] = np.asarray(adata.obsm[coords], dtype=float)
ha = HarremanAnalysis(adata, layer_key="counts")         # optionally model=<trained DestVI, ResolVI or SCVIVA>
ha.setup(compute_neighbors_on_key=coords, species="human", database="both", n_neighbors=...)  # species: "human" or "mouse"
ha.filter_genes(model="danb", autocorrelation_filt=True)
ha.compute_gene_pairs(ct_specific=False)                 # default True expects cell-type mode
ha.compute_cell_communication(model="danb", test="parametric")   # default test="both" also runs 1000 permutations
ha.select_significant_interactions(fdr_threshold=0.05, test="parametric")
sig = ha.results.significant_interactions                # {"gp": gene-pair DataFrame, "m": metabolite DataFrame}
gp = sig["gp"]; print(gp.columns.tolist())               # includes "Gene 1", "Gene 2", "Z", ...
top = gp.sort_values("Z", ascending=False)
```

- Steps must run in this order: `setup`, then `filter_genes`, then `compute_gene_pairs`, then
  `compute_cell_communication`, then `select_significant_interactions`.
- Cell-type-aware mode: pass `cell_type_key=` to `setup`, then use
  `compute_cell_communication(mode="cell_type")`. The results go to
  `ha.results.ct_cell_communication`.
- All results are stored in `adata.uns`. `ha.results` is a typed view of them
  (`autocorrelation`, `gene_pairs`, `cell_communication`, `interacting_cell_scores`, ...).
