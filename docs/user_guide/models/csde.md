# CSDE

**CSDE** (`scviva.tl.csde`, `scviva.tl.CSDEAnalysis`) — Corrected Spatial Differential
Expression — corrects for systematic errors from automated spatial transcriptomics pipelines
(cell mis-segmentation, mislabeling) before they propagate into false discoveries, by combining
a large automated-annotation set with a small manually-validated subset via prediction-powered
inference (PPI) to recover unbiased differential-expression estimates with valid confidence
intervals.

The advantages of CSDE include:

-   Corrects both the point estimate and the confidence interval for DE between two spatial
    populations of the same cell type, using a manually-validated subset to de-bias automated
    labels/segmentation.
-   Poisson or negative-binomial noise models.
-   Supports importance-weighted validated subsets (e.g. deliberately oversampled rare cell
    types), reweighting them back to the true population proportion.

The limitations of CSDE include:

-   Requires a manually-validated ground-truth subset — this scviva-tools integration reads
    that subset (produced by upstream CSDE's annotation workflow) but does not generate it; see
    "What's not (yet) in scviva-tools" below.
-   Compares exactly two spatial populations of one cell type at a time (e.g. "inside tumour"
    vs. "outside tumour" macrophages).

```{topic} Tutorials:

-   {doc}`/tutorials/CSDE_tutorial`
```

```{topic} External links:

-   [CSDE GitHub](https://github.com/YosefLab/CSDE)
-   [CSDE preprint](https://doi.org/10.64898/2026.01.15.699786)
```

## Overview

Upstream CSDE runs as three sequential steps. **Only step 3 is implemented in scviva-tools:**

1. **Export annotation panels** (upstream `scripts/export.py`, not in scviva-tools) —
   importance-samples a small subset of cells (e.g. 600) and renders a per-cell image panel
   (fluorescence crop + transcript dots + top-gene bar chart) for manual review.
2. **Manual validation** (upstream `scripts/annotate.py`, a Streamlit UI, not in scviva-tools) —
   a human annotator marks each exported cell as correctly or incorrectly
   segmented/labeled, producing `annotations.json`.
3. **Differential expression** (`scviva.tools.csde.tl.run_csde` /
   {class}`~scviva.tl.CSDEAnalysis`, **implemented in scviva-tools, ported from JAX to
   PyTorch**) — fits a per-class-intercept Poisson or negative-binomial model via
   prediction-powered inference, combining the validated subset with the full automated
   population to produce corrected log-fold-change estimates, p-values, and BH-adjusted
   q-values.

{meth}`~scviva.tl.CSDEAnalysis.from_spatialdata` reads a SpatialData object's `"table"`
AnnData together with an upstream annotation directory (`config.json`, `metadata.csv`,
`annotations.json`) to build the two population's inputs directly. `CSDEAnalysis`'s base
constructor only needs plain `AnnData` objects and column names, so it also composes directly
with any other scviva-tools model's output written into `.obs`/`.layers` — e.g. a scANVI
`cell_type` prediction, a scVIVA niche assignment used as the spatial-group split, or a
ResolVI-denoised counts layer.

## What's not (yet) in scviva-tools

The panel-export and manual-validation UI (steps 1-2 above) are not ruled out for a future
scviva-tools iteration — they're out of scope for the current one so this increment stays
focused. If picked up later, that work should get its own design doc and likely reuse
scviva-tools' existing plotting conventions (`scviva.pl.*`) rather than a 1:1 Streamlit port.
Until then, run steps 1-2 from the upstream [YosefLab/CSDE](https://github.com/YosefLab/CSDE)
repository directly; its output plugs straight into `CSDEAnalysis.from_spatialdata()`.
