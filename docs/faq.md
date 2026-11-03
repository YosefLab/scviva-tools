# Frequently Asked Questions

## Installation

### Which Python versions are supported?

scviva-tools requires Python 3.12 or later.

### Can I install scviva-tools without GPU support?

Yes. The base package runs entirely on CPU. GPU acceleration is available via the optional
`rapids` extra: `pip install "scviva-tools[rapids]"`.

## Data preparation

### What format does my data need to be in?

All models expect raw count data stored in an `AnnData` object. Normalized or log-transformed
data will cause numerical issues. Pass the correct `layer` argument to `setup_anndata` if your
counts are not in `adata.X`.

### What is the difference between `batch_key` and `categorical_covariate_keys`?

`batch_key` registers the primary technical covariate (e.g. sequencing run, donor slice) and
unlocks model features such as per-batch dispersion and counterfactual decoding.
`categorical_covariate_keys` accepts multiple secondary covariates but supports fewer downstream
features. Prefer `batch_key` when you have a single dominant technical effect.

## Training

### My model produces NaN losses — how do I fix this?

Common causes:

- **Unnormalized input**: ensure `adata.X` (or the specified `layer`) contains raw integer counts.
- **Low-count cells/spots**: filter cells with very few total counts before training.
- **Learning rate too high**: try reducing `lr` by an order of magnitude.
- **Batch size too small**: small batches increase gradient variance; try `batch_size=256` or larger.

### How long should I train?

Monitor ELBO convergence in the training progress bar. Most models converge within 200–500 epochs
on typical datasets. ResolVI requires more epochs (default 50 is a starting point — increase for
larger datasets).

## Spatial analysis

### Do I need squidpy installed?

squidpy is required for the default `backend="squidpy"` in `compute_neighbors`. Install it with
`pip install "scviva-tools[spatial]"`. A RAPIDS GPU backend is also available.

### Can I use SpatialData objects directly?

Yes. All models expose `setup_spatialdata()` and `from_spatialdata()` class methods that accept
a `SpatialData` object and an optional `region` filter.

## Saving and loading

### How do I save and reload a trained model?

```python
model.save("my_model_dir/")
model = MyModel.load("my_model_dir/", adata=adata)
```

See {doc}`user_guide/use_cases/saving_and_loading_models` for full details.
