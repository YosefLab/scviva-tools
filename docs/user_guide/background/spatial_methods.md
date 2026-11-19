# Spatial Transcriptomics Methods

## Technology overview

Spatial transcriptomics (ST) measures gene expression while preserving the spatial position of
cells or spots within the tissue. Key platforms include:

| Platform          | Resolution                | Typical use case       |
| ----------------- | ------------------------- | ---------------------- |
| Visium (10x)      | Multi-cell spots (~55 µm) | Whole-tissue profiling |
| Xenium / MERSCOPE | Single-cell resolved      | High-plex FISH         |
| Slide-seq         | Near single-cell          | Broad coverage         |

## Key challenges

- **Spot deconvolution** (Visium): multiple cell types per spot → DestVI.
- **Segmentation noise** (resolved ST): transcript assignment errors → ResolVI.
- **Niche modelling**: capturing cellular microenvironment effects → scVIVA.

## Neighbour graphs

Spatial neighbour graphs encode tissue topology. scviva-tools computes these via
`model.compute_neighbors()` using squidpy (CPU) or RAPIDS (GPU). The resulting
`index_neighbor` and `distance_neighbor` arrays in `adata.obsm` are consumed by ResolVI
and scVIVA during training.

## SpatialData integration

All models support [SpatialData](https://spatialdata.scverse.org) objects via
`setup_spatialdata()` and `from_spatialdata()`.
