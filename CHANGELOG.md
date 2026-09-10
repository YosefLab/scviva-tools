# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Native Apple Silicon (MPS) support for scviva-tools' own harreman/hotspot code.**
  `scviva.utils.resolve_device` now resolves `device="auto"` to `mps` (previously
  cuda→cpu only, so Mac GPUs were silently ignored). Since MPS has no float64 kernel
  support at all (a Metal hardware limitation, not a version gap), every
  local-correlation/-autocorrelation and cell-cell-communication routine that used to
  hard-code `torch.float64` now uses the new `scviva.utils.stats_dtype(device)` helper,
  which returns `float32` on `mps` and keeps `float64` on `cpu`/`cuda`. Models that
  train via scvi-tools' Lightning trainer (GimVI, DestVI, ResolVI, DiagVI, Stereoscope,
  Tangram, SCVIVA's NicheVAE) can train on `mps` once scvi-tools resolves
  `accelerator="mps"` — either passed explicitly, or via `accelerator="auto"` when the
  `SCVI_ALLOW_MPS_AUTO=1` environment variable is set. scvi-tools' own `auto` resolution
  deliberately keeps defaulting to `cpu` otherwise (see `scvi.model._utils.parse_device_args`)
  so existing users' behavior doesn't silently change; the new `test_mps.yaml` CI workflow
  (below) sets that variable so these models are actually exercised on `mps` in CI, not just
  the harreman/hotspot code. Verified end-to-end training on MPS for ResolVI and SCVIVA
  specifically, since both sample raw `torch`/`pyro`
  Gamma/Dirichlet/Poisson distributions (native MPS kernels as of torch 2.14) outside
  of `scvi.distributions`' own MPS guards. Also fixed `ResolVI.compute_dataset_dependent_priors`
  returning numpy `float64` scalars (`background_ratio`, `median_distance`, etc.) that
  `RESOLVAE` registers as buffers via `torch.tensor(...)`; on MPS this silently inherited
  float64 and crashed on `.to("mps")` during module setup — now cast to Python `float`,
  matching the existing `float(...)` idiom already used for
  `downsample_counts_mean`/`std` in `_resolvae.py`. Full test suite (271 tests) passes
  identically on CPU and MPS.

- Dedicated **MPS CI workflow** (`.github/workflows/test_mps.yaml`), mirroring scvi-tools'
  own `test_mps.yaml`: runs the full test suite on a real Apple Silicon GitHub-hosted
  runner (`macos-14`) with `SCVI_ALLOW_MPS_AUTO=1`, `PYTORCH_ENABLE_MPS_FALLBACK=0`, and
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` so `accelerator="auto"`-trained models actually
  exercise `mps` instead of silently staying on `cpu`. Adds the corresponding
  `test-metal` hatch environment (`metal` + `spatial` features) to `pyproject.toml`,
  mirroring the existing `test-cuda` env. Gated the same way as `test_macos.yaml`/
  `test_cuda.yaml`: PR label (`mps tests`/`all tests`), daily schedule, or manual dispatch.

- **DiagVI** model (`scviva.external.diagvi.DIAGVI`), migrated from scvi-tools, for
  cross-modality alignment of paired spatial transcriptomics/proteomics data via a
  shared latent space with graph-guided and optimal-transport (Sinkhorn) losses;
  ships with two new tutorials (`DiagVI_spatial_transcriptomics`,
  `DiagVI_spatial_proteomics`). Its optional `geomloss`/`torch-geometric` dependencies
  are covered by the existing `scviva-tools[spatial]` extra.
- Shared `CyclicMultiDataLoader` (`scviva.model.utils._dataloaders`), consolidating
  the near-identical cyclic-batch dataloader previously duplicated between GIMVI and
  DiagVI.
- **VISION** signature-scoring and differential-expression toolkit
  (`scviva.tools.vision.VisionAnalysis`), ported from `visionpy`/R VISION
  :cite:p:`DeTomaso19` into scviva-tools with a subpackage layout mirroring
  Harreman's (`preprocessing/`, `tools/`, `phylo/`): per-cell signature
  scoring, KNN-graph autocorrelation (Geary's C), one-vs-all/one-vs-one
  differential expression, gene-importance ranking, signature clustering, and
  micro-clustering/meta-cluster pooling. Ships with a standalone
  `Vision_tutorial` notebook, a new VISION section in `DestVI_tutorial`, and
  a `docs/user_guide/models/vision.md` model page.
- **Harreman-VISION integration** (`scviva.tools.harreman.vision`,
  `HarremanAnalysis.vs`), letting VISION signature scores be computed on top
  of an existing Harreman KNN graph and cross-referenced against Harreman's
  Hotspot gene modules (`integrate_vision_hotspot_results`).
- `scviva.plotting.vision`: VISION-specific plotting functions, split out of
  `scviva.plotting.harreman` to mirror the `tools/vision` vs `tools/harreman`
  split.

### Changed

- **DiagVI** now built on `SpatialBaseModel` (`setup_spatialdata()`, RAPIDS
  backend support) instead of a standalone `BaseModelClass`.
- Docs theme: the default Read the Docs search panel was replaced with the
  scviva-tools logo; project scaffolding updated to scverse cookiecutter v0.8.
- CI: GitHub Actions workflows updated for Python 3.14; CUDA workflows no
  longer cache pip installs, avoiding repeated 10GB cache-quota evictions.
- **Docs builds now treat warnings as errors** (`sphinx-build -W`), enforced both on
  Read the Docs (`sphinx.fail_on_warning`) and in a new CI `docs` job, mirroring
  scvi-tools' own doc-build hardening. `myst_heading_anchors` enabled so in-page
  `[text](#anchor)` links resolve; deprecated `logo_only` theme option removed;
  PyTorch's intersphinx inventory URL updated to `docs.pytorch.org`.

### Fixed

- **VISION R-fidelity audit**: correctness and reproducibility fixes found by
  auditing the port against the original R VISION implementation:
  - Protein one-vs-all differential results were computed but discarded (only
    the storage key was written); now persisted
  - Degenerate two-site contingency tables now report "not significant" like
    R's `matrix_chisq`, instead of maximal significance
  - Gene importance now uses covariance (matching R's
    `evalSigGeneImportance`/`sigGeneInner`) instead of a scale-free Pearson
    correlation
  - Added R's Wilcoxon continuity correction to one-vs-all differential
    expression, with a defensive `p<=1` clip
  - Meta-variable one-vs-all differential now applies one joint BH-FDR across
    numeric and categorical tests per group, matching `clusterSigScores`
  - Signature clustering rewritten as a BIC-selected Gaussian mixture over
    significant signatures (approximating R's `Mclust` "EII"), replacing
    plain Ward/Euclidean clustering
  - `sig_norm_method` now threads into the Geary's C permutation-null
    background scoring (previously hardcoded), removing an unnormalized
    raw-matmul code path
  - Micro-clustering's `K`/`filter_threshold` defaults corrected to match R's
    actual formulas (5% of cells, no artificial floor)
  - `pool_metadata` now reports `0.0` (not `NaN`) for categorical levels a
    pool doesn't contain, matching R's `poolMetaData`
  - Fixed a `visionpy`-inherited bug in the dense-matrix path of
    `compute_signatures_anndata`: the z-score denominator divided variance by
    gene count before the square root, inflating scores by an extra
    `sqrt(n_genes)` factor relative to R and this module's own sparse path
  - `norm_data_key` input is now auto-detected as already log-transformed
    (and left alone) instead of being re-transformed; docstring corrected (R's
    actual contract is the opposite of what was documented)
  - `device="cpu"` now forces the deterministic scipy sparse path instead of a
    non-bit-reproducible multi-threaded dense PyTorch path; Louvain
    clustering, K-means readjustment, and permutation-null background
    generation are now all seeded via new `random_state` parameters (default
    `0`, so existing calls are unaffected)
- **VISION correctness**: fixed BH-FDR index scrambling and tie-correction/NaN
  bugs in `diffexp`, KNN self-loop bias, and several silent
  `use_raw`/signature-splitting regressions
- **VISION**: `_infer_obs_columns()` no longer treats all-unique
  non-categorical columns (e.g. barcode/identifier columns) as usable
  metadata, which previously crashed differential expression on real
  metadata; `compute_signature_scores()` now restores
  `adata.uns["signature_varm_key"]` after the permutation-null step instead of
  leaving it pointed at the transient random-signature set (which broke
  `integrate_vision_hotspot_results()`); `VisionAnalysis` now tracks whether
  it built `adata.obsp["weights"]` itself, so it correctly reuses an
  externally-built graph (e.g. from Harreman) instead of silently overwriting
  it
- **VISION**: `import scviva` no longer eagerly pulls in `diffexp` (and
  therefore scanpy); `rank_genes_groups` is now exposed lazily.
  `HarremanAnalysis.vs.compute_vision_signatures()` now has explicit defaults
  instead of forwarding `**kwargs`, fixing the no-arg legacy call path;
  `load_signatures()` now accepts a `varm_key` and writes to
  `adata.raw.varm[varm_key]` under `use_raw=True`, fixing a raw/non-raw
  gene-count mismatch crash; small (`<=5`) signature sets now get correctly
  aligned random-background cluster labels
- **VISION**: `compute_obs_df_scores()`/`_gearysc_for_dataframe()` no longer
  raise `ZeroDivisionError` on constant (zero-variance) numeric `obs`
  columns; these are now skipped and reported as "no detectable
  autocorrelation"
- **Harreman**: fixed `use_raw` crashes and silent bugs (`.raw.obs`,
  gene-subset views into `.raw.X`, a database-storage key mismatch) and
  unseeded RNGs across the hotspot, cell-communication, and KNN modules
- **DiagVI**: corrected reference-batch/libsize imputation, `mapping_df`
  collisions, and unlabeled-cell leakage into the classifier loss
- 31 pre-existing Sphinx warnings uncovered by the new `-W` build, all docs/type-hint
  only with no runtime behavior change:
  - Stale `:doc:` tutorial cross-refs in `DestVI`/`GIMVI`/`SCVIVA`/`RNAStereoscope`/
    `SpatialStereoscope`/`Tangram` docstrings pointed at a scvi-tools-style
    `tutorials/notebooks/...` path that doesn't match this repo's flat
    `docs/tutorials/` layout; also dropped a reference to an R tutorial that was
    never ported here
  - Malformed numpydoc bullet lists (missing blank line before/after) in
    `nicheVAE`'s and `SpatialPredictiveMixin`'s docstrings
  - `np.array`/`Union[np.array, None]` type annotations corrected to `np.ndarray`/
    `np.ndarray | None` in `_de_utils.py`/`_results_dataclass.py`; stray quote
    removed from a docstring summary line
  - Missing `Ingelfinger25` (CytoVI) BibTeX entry, copied from upstream scvi-tools'
    `references.bib`
  - `RESOLVAE`/`MRDeconv` removed from the `docs/api/developer.md` autosummary
    listing — these classes don't exist yet (tracked as future work), so
    autosummary failed to import them
  - `docs/architecture/index.md` added to the developer-docs toctree (was orphaned)
  - Cross-project refs to scvi-tools' own MrVI/CytoVI/scVI docs converted from
    broken local `{doc}` roles to plain external links
  - Three graphical-model SVGs referenced by `diagvi.md`/`stereoscope.md` were
    missing from this repo; imported from upstream scvi-tools'
    `docs/user_guide/models/figures/` into the same path here. One broken ref to a
    never-written `counterfactual_prediction` background page dropped

### Removed

- Confirmed-dead VISION code with no remaining call sites, found via a
  post-audit sweep: `tools/diffexp.py::_calc_frac()` (logic already computed
  inline elsewhere) and `_analysis.py::_pearson_cols()` (orphaned by the
  gene-importance fix above).

## [0.1.6] - 2026-07-08

### Fixed

- **Harreman**: pandas 3.0 always-on copy-on-write broke three in-place mutation sites
  that relied on `.values` returning a writable view:
  - `create_modules` raised `ValueError: underlying array is read-only` on
    `np.fill_diagonal`; now forces a real copy via `to_numpy(copy=True)`
  - `run_cell_communication_analysis` hit the same `fill_diagonal` failure on the
    symmetric LC Z-score matrix; now mutates a copy and writes it back via `.loc[:, :]`
  - `extract_lr_pairs` raised `ValueError: Length of indexer and values mismatch`
    because pandas 3.0's default Arrow-backed string dtype no longer allows storing
    variable-length arrays into single-element slots; ligand/receptor columns now use
    `to_numpy(dtype=object, copy=True)` to restore plain, guaranteed-writable
    object-array semantics

## [0.1.5] - 2026-07-06

### Added

- **Tangram** model for mapping scRNA-seq to spatial transcriptomics
  (`scviva.external.Tangram`), a PyTorch reimplementation supporting "cells" and
  "constrained" cell-to-spot mapping modes
- **Harreman** downstream tool for cell-cell communication and metabolic exchange
  analysis (`scviva.tl.harreman`, `scviva.pl.harreman`), integrating outputs from
  DestVI, ResolVI, and SCVIVA
- **`SpatialPredictiveMixin`**: unified predictive mixin shared by ResolVI and SCVIVA,
  replacing the ResolVI-only `ResolVIPredictiveMixin`
- **`scviva.tl`/`scviva.pl`** top-level namespace aliases, mirroring scanpy's
  `sc.tl`/`sc.pl` convention
- Dedicated CI job for optional/slow integration tests (real model training)

### Changed

- **Package renamed** from `spatialvi-tools` to `scviva-tools` on PyPI (import name
  `scviva` unchanged); docs and install instructions updated accordingly

### Fixed

- **Tangram**: gene-mismatch validation in `setup_mudata`, and `get_mapper_matrix`
  filtering under `constrained=True`
- **Harreman**: import/namespace bugs from the initial port that made the package
  unimportable, plus a broken plotting accessor and a model-recognition constant typo

## [0.1.3] - 2026-04-13

### Changed

- **Cleanup**: dropped dead code (MLflow/JAX settings, a stale config re-export, a
  trivial method alias, stale in-source dev comments) and simplified the lazy model map
- **Refactored** `SCVIVA`'s predictive methods to share a common decoder helper, and
  tidied up several module-level imports; moved a few internal modules under `utils/`
  for consistency with scvi-tools conventions

### Fixed

- `scikit-learn>=1.4` moved from test-only to core dependencies (required by SCVIVA/ResolVI
  at runtime); removed redundant `anndata`/`scanpy` test-extra pins

### Documentation

- Added installation guide, FAQ, API reference stubs, developer/contributing guides, and
  user-guide background/use-case pages

## [0.1.2] - 2026-04-12

### Fixed

- `RNADeconv.__init__`: read `model_kwargs['ct_weight']` instead of the incorrect
  `model_kwargs['ct_prop']`; the mismatched key caused a `KeyError` whenever
  `ct_weight` was passed, making class-reweighted training unusable
- `JVAE.generative`: replace `y.type.squeeze(-1)` with `y.squeeze(-1).long()` for the
  gene-label dispersion branch; `y.type` is Python's built-in `type()` bound on the
  tensor object, so any run with `dispersion='gene-label'` crashed before reaching
  the `one_hot` encoding
- Added regression tests for both fixes (all 78 tests pass)

## [0.1.1] - 2026-04-12

### Added

- **GIMVI** model for joint imputation of missing genes across paired scRNA-seq and
  spatial transcriptomics data, based on :cite:p:`Lopez19`
- **Stereoscope** model for probabilistic cell-type deconvolution of spatial
  transcriptomics spots (`spatialvi.external.Stereoscope`)
- Tutorials: `gimvi_tutorial`, `stereoscope_heart_LV_tutorial`,
  `cell2location_lymph_node_spatial_tutorial`, and `tangram_scvi_tools`
- Regression test suites for GIMVI and Stereoscope (upstream parity tests)

## [0.1.0] - 2026-03-25

### Added

- Initial package with DestVI, ResolVI, and scVIVA models
- `SpatialBaseModel` shared base infrastructure
- `SpatialNeighborhoodMixin` for neighbor graph computation (squidpy + RAPIDS backends)
- `SpatialDeconvolutionMixin` for cell-type deconvolution
- SpatialData integration via `setup_spatialdata()` and `from_spatialdata()`
- squidpy neighbor computation backend
- RAPIDS acceleration backend (`backend="rapids"`)
- Custom AnnData fields: `SpatialCoordsField`, `NeighborhoodGraphField`
