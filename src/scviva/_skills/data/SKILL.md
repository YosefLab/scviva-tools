---

name: scviva
description: "Use when analysing spatial transcriptomics with the scviva-tools package (import scviva): cell-type deconvolution of spots from a scRNA-seq reference (DestVI, Stereoscope, Tangram), niche-aware embeddings and niche differential expression (scVIVA), denoising or relabelling cells and DE (ResolVI), gene imputation (GIMVI), or spatial cell-cell and metabolic communication (Harreman)."
---

# scviva-tools (`import scviva`)

Spatial models built on scvi-tools. `scvi-tools` is installed too, and some classes exist
in both packages. **Use the `scviva` versions.** The `scvi.*` copy of DestVI does not work
with the installed versions (details below).

## Running work: the run is graded when your turn ends

- Training takes minutes. Run each script in the **foreground** with a long Bash timeout
  (`timeout=600000`). Always pass an explicit `max_epochs`. `scviva.DestVI.train` defaults
  to 2000 epochs.
- If a job may run past one call, split it into steps. Save each trained model with
  `model.save(dir, overwrite=True)` and load it in the next step with
  `Cls.load(dir, adata=adata)`.
- Write the result file (e.g. `answer.md`) from inside the script as soon as the result exists.
  **Never end your turn while a job is still running in the background.** Wait for it to
  finish and check the output file before you stop.
- A CUDA GPU is usually available, so keep `accelerator="auto"`. Do not force CPU unless you
  have to. CPU training is much slower and is the usual reason a run times out.

## Pick the entry point by goal

| Goal                                                                           | Entry point                                                                                                        | Recipe                     |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| Cell-type proportions per spot from a scRNA reference                          | `scvi.model.CondSCVI` then `scviva.DestVI.from_rna_model`                                                          | references/destvi.md       |
| Niche-aware latent space, or DE between groups that accounts for neighbourhood | `scviva.SCVIVA`                                                                                                    | references/scviva_niche.md |
| Correct or reassign cell labels, denoise, DE on cellular-resolution data       | `scviva.ResolVI` (same as `scviva.model.ResolVI`)                                                                  | references/resolvi.md      |
| Ligand-receptor or metabolite-transport communication                          | `scviva.tools.harreman.HarremanAnalysis`                                                                           | references/harreman.md     |
| Other deconvolution or mapping methods, or imputation                          | `scviva.external.RNAStereoscope` + `scviva.external.SpatialStereoscope`, `scviva.external.Tangram`, `scviva.GIMVI` | see the class docstrings   |

## Traps that cost runs

- **DestVI:** use `scviva.DestVI`, **not** `scvi.model.DestVI`. The scvi copy raises
  `KeyError: 'prior'`. With patches it then raises "cannot assign numpy.ndarray to buffer
  mean_vprior", and after that a cuda/cpu device mismatch. Agents lost whole runs patching
  around these errors. `scviva.DestVI.from_rna_model(st_adata, condscvi_model)` works unpatched
  with both CondSCVI priors (default `"normal"` and `"mog"`), on GPU.
- `from_rna_model` registers the spatial AnnData itself, copying the CondSCVI setup
  arguments. If CondSCVI used a `batch_key`, that `obs` column must also exist in the spatial
  AnnData, or you get `KeyError: '<col> not found in adata.obs'`. Subset both AnnData objects
  to the same genes before training CondSCVI.
- **SCVIVA:** call `SCVIVA.preprocessing_anndata(...)` **before** `setup_anndata`, with the same
  keys. It needs an expression embedding already in `obsm` (e.g. `X_scVI` or `X_scANVI`).
  `differential_expression` returns a **dataclass**, not a DataFrame. The main table is
  `res.g1_g2`.
- **ResolVI:** `setup_anndata` builds spatial neighbours from `obsm["X_spatial"]` by default.
  If the coordinates are stored under another key, pass
  `prepare_data_kwargs={"spatial_rep": "<key>"}`. `predict(soft=False)` returns a numpy
  array of label strings.
- **Harreman:** use `compute_gene_pairs(ct_specific=False)` for the standard (not
  cell-type-aware) analysis. Use the same `test=` in `compute_cell_communication` and
  `select_significant_interactions`. Drop cells with NaN coordinates first.
- When a task specifies a parameter (epochs, delta, FDR, k, test mode, prior…), pass it
  explicitly. The defaults here often differ from scvi-tools. When it does not, use the defaults.
