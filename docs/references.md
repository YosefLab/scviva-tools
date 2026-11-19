# References

BibTeX entries are in [`references.bib`](references.bib).

---

## Core Models

### scVIVA

> **Levy et al. (2025)**
> _scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics_
> bioRxiv. doi: [10.1101/2025.06.01.657182](https://doi.org/10.1101/2025.06.01.657182)
> {cite:p}`Levy25`

scVIVA models gene expression as a function of both the cell's intrinsic state and its
microenvironment (niche). Niche-conditioned decoders disentangle cell-intrinsic from
environment-driven variation, enabling niche-aware differential expression.

### DestVI

> **Lopez et al. (2022)**
> _DestVI identifies continuums of cell types in spatial transcriptomics data_
> _Nature Biotechnology_. doi: [10.1038/s41587-022-01272-8](https://doi.org/10.1038/s41587-022-01272-8)
> {cite:p}`Lopez22`

DestVI performs multi-resolution deconvolution of spatial transcriptomics spots into
cell-type compositions using a conditional SCVI reference trained on scRNA-seq.

### ResolVI

> **Ergen & Yosef (2025)**
> _ResolVI - addressing noise and bias in spatial transcriptomics_
> bioRxiv. doi: [10.1101/2025.01.20.634005](https://doi.org/10.1101/2025.01.20.634005)
> {cite:p}`Ergen25`

ResolVI corrects segmentation errors, background signal, and cell-size bias in
cellular-resolution spatial transcriptomics (Xenium, MERFISH, CosMx) using a
Pyro-based probabilistic model with neighbor-aware decoders.

### Harreman

> **Etxezarreta Arrastoa et al. (2025)**
> _Metabolic zonation and characterization of tissue slices with spatial transcriptomics_
> bioRxiv. doi: [10.1101/2025.11.11.687271](https://doi.org/10.1101/2025.11.11.687271)
> {cite:p}`Etxezarreta-Arrastoa25`

Harreman infers spatially-resolved metabolic gene programs and cell-cell metabolic/ligand-receptor
communication from spatial transcriptomics data using local autocorrelation and spatial proximity graphs.

### VISION

> **DeTomaso et al. (2019)**
> _Functional interpretation of single cell similarity maps_
> _Nature Communications_, 10, 4376. doi: [10.1038/s41467-019-12235-0](https://doi.org/10.1038/s41467-019-12235-0)
> {cite:p}`DeTomaso19`

VISION scores curated gene signatures on a per-cell basis and tests them for autocorrelation
across a cell-cell neighbor graph, enabling interpretation of single-cell similarity maps.

### CSDE

> **Boyeau et al. (2026)**
> _Corrected spatial differential expression via prediction-powered inference_
> bioRxiv. doi: [10.64898/2026.01.15.699786](https://doi.org/10.64898/2026.01.15.699786)
> {cite:p}`Boyeau26`

CSDE corrects for cell mis-segmentation and mislabeling errors from automated spatial
transcriptomics pipelines before they propagate into false discoveries, by combining a large
automated-annotation set with a small manually-validated subset via prediction-powered
inference to recover unbiased differential-expression estimates with valid confidence intervals.

---

## Foundation: scvi-tools

> **Gayoso et al. (2022)**
> _A Python library for probabilistic analysis of single-cell omics data_
> _Nature Biotechnology_, 40, 163–166. doi: [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w)
> {cite:p}`Gayoso22`

> **Lopez et al. (2018)**
> _Deep generative modeling for single-cell transcriptomics_
> _Nature Methods_, 15, 1053–1058. doi: [10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2)
> {cite:p}`Lopez18`

---

## scverse Ecosystem

> **Virshup et al. (2024)** — The scverse project
> _Nature Biotechnology_, 42, 333–336. doi: [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8)
> {cite:p}`Virshup24`

> **Palla et al. (2022)** — squidpy
> _Nature Methods_, 19, 171–178. doi: [10.1038/s41592-021-01358-2](https://doi.org/10.1038/s41592-021-01358-2)
> {cite:p}`Palla22`

> **Wolf et al. (2018)** — scanpy
> _Genome Biology_, 19, 15. doi: [10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0)
> {cite:p}`Wolf18`

---

## Deep Learning Frameworks

> **Paszke et al. (2019)** — PyTorch. NeurIPS 2019. {cite:p}`Paszke19`

> **Falcon et al. (2019)** — PyTorch Lightning. {cite:p}`Falcon19`

> **Bingham et al. (2019)** — Pyro. _JMLR_, 20(28). {cite:p}`Bingham19`

---

## GPU Acceleration

> **NVIDIA Corporation** — RAPIDS (cuML / cuGraph). https://rapids.ai {cite:p}`RAPIDS`

---

## Future External Models

| Model                 | Reference                                            | Key                        |
| --------------------- | ---------------------------------------------------- | -------------------------- |
| Stereoscope           | Andersson et al. (2020), _Commun Biol_               | {cite:p}`Andersson20`      |
| Tangram               | Biancalani et al. (2021), _Nature Methods_           | {cite:p}`Biancalani21`     |
| Cell2location         | Kleshchevnikov et al. (2022), _Nature Biotechnology_ | {cite:p}`Kleshchevnikov22` |
| starfysh              | Chang et al. (2023) — BibTeX entry TBD               | —                          |
| VIVS / SPARL / others | References TBD                                       | —                          |

---

## Bibliography

```{bibliography}
:cited:
```
