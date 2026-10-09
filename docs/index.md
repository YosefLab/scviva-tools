# scVIVA-Tools

**scVIVA-Tools** is a consolidated spatial transcriptomics analysis toolkit built on top of [scvi-tools](https://scvi-tools.org), exposing ResolVI, DestVI, scVIVA, DiagVI, Amici, Starfysh, Harreman, gimVI, Stereoscope, and Tangram through a clean, unified API.

```{toctree}
:maxdepth: 1
:hidden:

installation
user_guide/index
tutorials/index
api/index
developer/index
faq
changelog.md
references
Discussion <https://discourse.scvi-tools.org>
GitHub <https://github.com/YosefLab/scviva-tools>
Model hub <https://huggingface.co/scvi-tools>
```

## Models

::::{grid} 3
:gutter: 2

:::{grid-item-card} ResolVI
:link: user_guide/models/resolvi
:link-type: doc

Denoising and segmentation-error correction for single-cell resolved spatial transcriptomics.
:::

:::{grid-item-card} DestVI
:link: user_guide/models/destvi
:link-type: doc

Cell-type deconvolution of spatial transcriptomics spots with sub-cell-type resolution.
:::

:::{grid-item-card} scVIVA
:link: user_guide/models/scviva
:link-type: doc

Niche-aware variational inference for cellular microenvironment modeling.
:::

:::{grid-item-card} DiagVI
:link: user_guide/models/diagvi
:link-type: doc

Diagonal integration of unpaired multi-modal single-cell data using prior cross-modal feature correspondences.
:::

:::{grid-item-card} Harreman
:link: user_guide/models/harreman
:link-type: doc

Metabolic exchange and cell-cell communication inference from spatial transcriptomics data.
:::

:::{grid-item-card} gimVI
:link: user_guide/models/gimvi
:link-type: doc

Joint imputation of missing genes across paired scRNA-seq and spatial transcriptomics datasets.
:::

:::{grid-item-card} Stereoscope
:link: user_guide/models/stereoscope
:link-type: doc

Two-stage deconvolution of cell type proportions in spatial transcriptomics spots.
:::

:::{grid-item-card} Tangram
:link: user_guide/models/tangram
:link-type: doc

Mapping single-cell RNA-seq data onto spatial transcriptomics via optimal transport.
:::

:::{grid-item-card} VISION
:link: user_guide/models/vision
:link-type: doc

Interpretation of single-cell similarity maps via annotated gene signature scoring.
:::

::::
