# Spatial Transcriptomics Methods

## Technology overview

Spatial transcriptomics (ST) measures gene expression while preserving the spatial position of
cells or spots within the tissue. Key platforms include:

| Platform | Resolution | Typical use case |
|----------|-----------|-----------------|
| Visium (10x) | Multi-cell spots (~55 µm) | Whole-tissue profiling |
| Xenium / MERSCOPE | Single-cell resolved | High-plex FISH |
| Slide-seq | Near single-cell | Broad coverage |

## Key challenges

- **Spot deconvolution** (Visium): multiple cell types per spot → DestVI.
- **Segmentation noise** (resolved ST): transcript assignment errors → ResolVI.
- **Niche modelling**: capturing cellular microenvironment effects → scVIVA.

## Neighbour graphs

Spatial neighbour graphs encode tissue topology. ResolVI consumes them as two arrays in
`adata.obsm`: `index_neighbor` holds, for each cell, its `n_neighbors` nearest cells
(excluding itself, sorted by distance), and `distance_neighbor` the corresponding
**squared** Euclidean distances. They can be produced in two ways:

- **Default:** `ResolVI.setup_anndata(adata)` (`prepare_data=True`) computes them per batch
  from `adata.obsm["X_spatial"]`, using scanpy, or RAPIDS SingleCell when it's installed.
  Pass `prepare_data_kwargs` (e.g. `{"n_neighbors": 15, "spatial_rep": "spatial"}`) to
  configure this.
- **Explicit:** `ResolVI.compute_neighbors(adata)`, called before `setup_anndata`, using
  squidpy (CPU) or RAPIDS (GPU). This computes neighbors over all cells, not per batch. A
  later `setup_anndata` keeps these neighbors instead of recomputing them, unless
  `prepare_data_kwargs` is passed.

Pre-computed neighbors from elsewhere can be used with `setup_anndata(adata,
prepare_data=False)`. They must follow the same convention (squared distances, self
excluded). scVIVA uses its own `niche_indexes`/`niche_distances` keys instead.

## SpatialData integration

All models support [SpatialData](https://spatialdata.scverse.org) objects via
`setup_spatialdata()` and `from_spatialdata()`.
