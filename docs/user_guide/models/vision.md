# VISION

**VISION** {cite:p}`DeTomaso19` (`scviva.tl.VisionAnalysis`) is a framework for interpreting single-cell (and spatial) transcriptomic similarity maps using annotated gene signatures.

The advantages of VISION include:

-   Unsupervised scoring of curated gene signatures (e.g. Hallmark, GO, custom) on a per-cell basis.
-   Spatial/graph autocorrelation (Geary's C) to identify signatures and metadata that vary systematically across the cell-cell neighbor graph.
-   One-vs-all differential expression of signatures and metadata across clusters, with per-signature gene importance and a signature dendrogram.
-   PhyloVision mode: builds the neighbor-weight graph from a phylogenetic tree instead of an expression/embedding latent space, and computes Fitch-Hartigan parsimony plasticity scores.

The limitations of VISION include:

-   Signature scoring requires curated gene sets (GMT files, R VISION `.txt` signature files, or plain dicts) — it does not discover signatures de novo (see {doc}`harreman` for unsupervised gene-module discovery).
-   The interactive web report from R VISION and visionpy (Flask app, REST routes, JS/HTML assets) was intentionally not ported; results are consumed programmatically via `va.results` or plotted directly from `adata`.

```{topic} Tutorials:

-   {doc}`/tutorials/Vision_tutorial`
-   {doc}`/tutorials/Visium_colon_Harreman_pipeline` (joint Harreman + VISION analysis via `ha.vs`)
```

```{topic} External links:

-   [R VISION documentation](https://yoseflab.github.io/VISION/)
-   [R VISION GitHub](https://github.com/YosefLab/VISION)
-   [visionpy GitHub](https://github.com/yoseflab/visionpy)
```

## Overview

VISION operates in up to four steps:

1.  **Setup** ({meth}`~scviva.tools.vision.VisionAnalysis.setup`): builds the cell-cell neighbor-weight graph (exponential kernel over a latent embedding, or over a phylogenetic tree in PhyloVision mode) and Louvain cell clusters.
2.  **Signature scoring** ({meth}`~scviva.tools.vision.VisionAnalysis.load_signatures`, {meth}`~scviva.tools.vision.VisionAnalysis.compute_signatures`): scores curated gene signatures per cell and computes their graph autocorrelation (Geary's C) against a permutation-based null.
3.  **Differential expression** ({meth}`~scviva.tools.vision.VisionAnalysis.compute_differential_expression`): one-vs-all Wilcoxon differential expression of signatures and metadata across clusters, per-signature gene importance, latent-component annotation, and a signature dendrogram/clustering.
4.  **PhyloVision (optional)**: passing `tree=` to `setup()` builds the neighbor graph from a Newick tree instead of an embedding; {meth}`~scviva.tools.vision.VisionAnalysis.tl.compute_plasticity_scores` (`va.tl.compute_plasticity_scores`) then scores how conserved each categorical metadata variable is along the lineage tree.

## Relationship to R VISION and visionpy

`scviva.tools.vision` is a direct Python port of [visionpy](https://github.com/yoseflab/visionpy), itself a port of the original [R VISION](https://github.com/YosefLab/VISION) package, excluding the interactive Flask web-report server. Where visionpy exposed its analysis session as a process-global singleton (`AnnDataAccessor`), `VisionAnalysis` instead follows {class}`~scviva.tools.harreman.HarremanAnalysis`'s pattern of an explicit, user-owned session object with an internal step-prerequisite gate (`va.setup()` before `va.compute_signatures()`, etc.) — the numerical results and `adata.obs`/`obsm`/`uns` key layout are otherwise unchanged from visionpy, so migrating an existing script is largely mechanical.

The port has been independently verified against the R source: it fixes three real R bugs (a PCA-loading row/column slice bug in permutation WPCA, an LCA-tree weight-matrix encoding inversion, and Fitch-Hartigan parsimony tie-break nondeterminism) while intentionally documenting two minor deviations (approximate KNN by default, where R VISION is always exact; no Wilcoxon continuity correction) that do not materially change results.

## Usage

```python
from scviva.tools.vision import VisionAnalysis

va = VisionAnalysis(adata)
va.setup(compute_neighbors_on_key="X_pca")
va.load_signatures(gmt_files=["h.all.v7.symbols.gmt"])
va.compute_signatures()
va.compute_differential_expression()

results = va.results
```

Without a precomputed latent space:

```python
va = VisionAnalysis(adata)
va.compute_latent_space()
va.setup()
```

VISION can also be run as part of a Harreman session on the same `adata`, reusing Harreman's spatial neighbor graph instead of building its own:

```python
from scviva import tl

ha = tl.HarremanAnalysis(adata)
ha.setup(compute_neighbors_on_key="spatial")
ha.vs.load_signatures(gmt_files=["h.all.v7.symbols.gmt"])
ha.vs.analyze_vision(signature_varm_key="signatures")
```

## API

Please see {mod}`scviva.tl.VisionAnalysis` for the full API reference.
