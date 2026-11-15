---

name: scviva
description: "Use when analyzing spatial transcriptomics with the scviva-tools package (import scviva): niche-aware embeddings and niche differential expression (SCVIVA), spot deconvolution into cell-type proportions (DestVI, Stereoscope, Tangram), denoising/relabeling cellular-resolution data (ResolVI), imputing unmeasured genes or integrating modalities (GIMVI, DIAGVI), and spatial cell-cell or metabolic communication and gene-signature scoring (Harreman, VISION)."
---

# scviva-tools (`import scviva`)

Spatial extensions of scvi-tools. The models use the usual scvi pattern
(`Model.setup_anndata(adata, ...)` → `Model(adata)` → `.train()` → getters). Harreman and VISION
are stateful analysis objects whose steps must run in order.

## Pick the entry point from the goal

| Goal                                                                                                                                   | Entry point                                                                                                 | Details                         |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------- |
| Niche-aware latent space, predict neighborhood composition, DE between cell groups that discounts contamination from spatial neighbors | `scviva.model.SCVIVA`                                                                                       | `references/scviva.md`          |
| Cell-type proportions per spot (Visium-like) from an scRNA reference                                                                   | `scviva.model.DestVI` (+ `scvi.model.CondSCVI`), or `scviva.external.RNAStereoscope` + `SpatialStereoscope` | `references/deconvolution.md`   |
| Map single cells onto spots, project labels or genes into space                                                                        | `scviva.external.Tangram` (MuData input)                                                                    | `references/deconvolution.md`   |
| Denoise cellular-resolution ST, (re)annotate cells semi-supervised, DE after correction                                                | `scviva.model.ResolVI`                                                                                      | `references/resolvi.md`         |
| Impute genes missing from a spatial panel using scRNA-seq                                                                              | `scviva.model.GIMVI`                                                                                        | `references/integration.md`     |
| Integrate unpaired modalities (e.g. RNA + spatial proteomics), cross-modal imputation or label transfer                                | `scviva.external.DIAGVI`                                                                                    | `references/integration.md`     |
| Ligand-receptor or metabolite exchange between neighboring cells or spots, spatially autocorrelated genes                              | `scviva.tools.HarremanAnalysis`                                                                             | `references/harreman_vision.md` |
| Score gene signatures per cell and test their autocorrelation on a cell graph                                                          | `scviva.tools.VisionAnalysis`                                                                               | `references/harreman_vision.md` |

Imports that work: `from scviva.model import SCVIVA, DestVI, ResolVI, GIMVI`,
`from scviva.external import Tangram, RNAStereoscope, SpatialStereoscope, DIAGVI`,
`from scviva.tools import HarremanAnalysis, VisionAnalysis`. GIMVI lives in `scviva.model`,
not `scviva.external`.

## Rules that apply to every model

- Train on raw counts. If `adata.X` is normalized or log-transformed, pass `layer="counts"`
  (whichever layer holds integers) to `setup_anndata`.
- Seed with `scviva.settings.seed = 0` before building models.
- Every key you name must exist, and several setup defaults point at keys your data probably
  lacks: SCVIVA defaults to `labels_key="cell_type"`, `expression_embedding_key="X_scVI"`,
  `cell_coordinates_key="spatial"`; ResolVI looks for coordinates in `obsm["X_spatial"]`.
  Inspect `adata.obs.columns` and `adata.obsm.keys()` first and pass the real names.
- If the task states hyperparameters (epochs, k, delta, FDR target, lr, batch size, test
  type), pass them exactly; they change the answer. Otherwise keep the defaults. On CPU the
  defaults can be slow (DestVI trains 2000 epochs); lower `max_epochs` only if the task allows.
- scvi-style `differential_expression` tables are sorted by `proba_de`, not by effect size.
  For "most up-regulated gene in group1", sort by `lfc_mean` descending (optionally restricted
  to rows where `is_de_fdr_<fdr_target>` is True). Positive `lfc_mean` means higher in `group1`.
  `group1` accepts a string or list; `group2=None` compares against all other cells.
- Getters return pandas objects indexed by `obs_names` and `var_names` (e.g. proportions are
  spots × cell types). Align with `adata.obs` by index before grouping.
