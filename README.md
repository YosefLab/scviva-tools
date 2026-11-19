# scviva-tools

Consolidated spatial transcriptomics analysis toolkit based on [scvi-tools](https://scvi-tools.org).

Provides seven spatial transcriptomics models and tools as first-class citizens:
- **scVIVA** — niche-aware representation learning (Levy et al., 2025)
- **DestVI** — multi-resolution cell-type deconvolution (Lopez et al., 2022)
- **ResolVI** — noise and bias correction for cellular-resolution ST (Ergen & Yosef, 2025)
- **GIMVI** — joint imputation of missing genes across paired scRNA-seq and spatial data (Lopez et al., 2019)
- **Stereoscope** — probabilistic cell-type deconvolution of spatial spots (Andersson et al., 2020)
- **Tangram** — mapping scRNA-seq data onto spatial coordinates (Biancalani et al., 2021)
- **Harreman** — metabolic exchange and cell-cell communication inference for spatial data (Etxezarreta-Arrastoa et al., 2025), integrating outputs from DestVI, ResolVI, and SCVIVA

See [docs/user_guide/models](docs/user_guide/models) for a full description of each model, and
[docs/tutorials](docs/tutorials) for worked examples of every model above.

## Installation

```bash
pip install scviva-tools

# Optional extras
pip install "scviva-tools[spatial]"    # SpatialData + squidpy integration
pip install "scviva-tools[rapids]"     # GPU-accelerated neighbor graphs (cuML/cuPy/cuGraph)
pip install "scviva-tools[tutorials]"  # jupyter, matplotlib, seaborn
pip install "scviva-tools[all]"        # everything above
```

See [docs/installation.md](docs/installation.md) for details, including a development install.

## Quick Start

```python
import scviva

# scVIVA
scviva.SCVIVA.setup_anndata(adata, layer="counts", spatial_key="spatial")
model = scviva.SCVIVA(adata)
model.train()

# DestVI
import scvi
scvi.model.CondSCVI.setup_anndata(sc_adata, labels_key="cell_type", layer="counts")
sc_model = scvi.model.CondSCVI(sc_adata)
sc_model.train()

scviva.DestVI.setup_anndata(st_adata, layer="counts")
st_model = scviva.DestVI.from_rna_model(st_adata, sc_model)
st_model.train()

# ResolVI
scviva.ResolVI.setup_anndata(adata, layer="counts", spatial_key="spatial")
model = scviva.ResolVI(adata)
model.train()
```

Models beyond scVIVA/DestVI/ResolVI (GIMVI, Stereoscope, Tangram) live under `scviva.external`,
e.g. `scviva.external.Tangram`.

### Downstream tools (`scviva.tl` / `scviva.pl`)

`scviva.tl` and `scviva.pl` mirror scanpy's `sc.tl`/`sc.pl` convention, resolving lazily to
`scviva.tools`/`scviva.plotting` so heavy optional dependencies aren't imported eagerly:

```python
# Harreman: metabolic exchange & cell-cell communication analysis, optionally
# integrating DestVI/ResolVI/SCVIVA outputs
scviva.tl.harreman.hs.compute_local_autocorrelation(adata, use_metabolic_genes=True)
scviva.pl.harreman.local_correlation_plot(adata)
```

`scviva.tl.harreman` itself exposes a scanpy-style sub-API: `tl`/`hs`/`pp`/`ds`/`pl` for
tools/hotspot/preprocessing/datasets/plotting, plus a stateful `HarremanAnalysis` wrapper
(`scviva.tools.harreman.HarremanAnalysis`) that can integrate DestVI/ResolVI/SCVIVA outputs.

## Documentation

Full documentation, including the user guide, API reference, and tutorials, is under
[docs/](docs/index.md). See [CHANGELOG.md](CHANGELOG.md) for a detailed history of recent
changes.

## Citation

If you use scviva-tools in your research, please cite:

> Ori Kronfeld, Nathan Levy, Pierre Boyeau, Oier Exezarreta-Arrastoa, Artemii Bakulin, Elham Azizi, Can Ergen & Nir Yosef

## References

See [docs/references.md](docs/references.md) for full citations.

Copyright (c) 2026, Yosef Lab, Weizmann Institute of Science
