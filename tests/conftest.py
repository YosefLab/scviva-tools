"""Shared pytest fixtures for scviva test suite."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model.base import UnsupervisedTrainingMixin

from scviva.data._fields import SpatialCoordsField
from scviva.model.base._spatial_base import SpatialBaseModel


def pytest_addoption(parser):
    """Add the --optional and --seed command line options."""
    parser.addoption(
        "--optional",
        action="store_true",
        default=False,
        help="Run tests marked as optional (e.g. requiring real model training).",
    )
    parser.addoption(
        "--seed",
        action="store",
        default=0,
        help="Option to specify which scvi-tools seed to use for tests.",
    )


@pytest.fixture(autouse=True)
def set_seed(request):
    """Sets the scvi-tools seed for each test, matching upstream scvi-tools' conftest."""
    from scvi import settings

    settings.seed = int(request.config.getoption("--seed"))


def pytest_fixture_post_finalizer(fixturedef):
    """Frees the MPS caching allocator's cached blocks after every fixture teardown.

    Unlike CUDA's dedicated VRAM, MPS memory is unified with host RAM, and the whole
    suite runs as one pytest process, so cached (but unused) tensor memory from each
    test's models/optimizers must be reclaimed promptly or it accumulates across the
    run and exhausts the shared pool, crashing unrelated later tests with "MPS backend
    out of memory".

    A function-scoped autouse fixture is not enough: pytest tears down function-scoped
    fixtures before higher-scoped ones, so on the last test of a module the cache would
    be purged *before* a module-scoped fixture (e.g. a trained model shared across the
    module) releases its tensors -- leaving that memory cached, unreclaimed, right as
    the next module's fixtures spin up. Hooking every fixture's finalizer instead (not
    just function-scoped ones) purges the cache the moment any fixture's tensors --
    regardless of scope -- actually become unreferenced.
    """
    del fixturedef
    import gc

    import torch

    if torch.backends.mps.is_available():
        # gc.collect() first: the caching allocator can only release blocks that are no
        # longer referenced by Python (e.g. a trainer/model/optimizer held by the fixture
        # that just tore down); empty_cache() alone leaves those blocks cached.
        gc.collect()
        torch.mps.empty_cache()


def pytest_collection_modifyitems(config, items):
    """Skip optional tests unless --optional is passed, and vice versa."""
    run_optional = config.getoption("--optional")
    skip_optional = pytest.mark.skip(reason="need --optional option to run")
    skip_non_optional = pytest.mark.skip(reason="test not marked with pytest.mark.optional")
    for item in items:
        if not run_optional and "optional" in item.keywords:
            item.add_marker(skip_optional)
        elif run_optional and "optional" not in item.keywords:
            item.add_marker(skip_non_optional)


class _MinimalSpatialModel(SpatialBaseModel, UnsupervisedTrainingMixin):
    """Minimal concrete model for testing SpatialBaseModel methods in isolation.

    Cannot be trained (no real module.inference). Used only for testing
    field registration, setup_spatialdata, and plotting helpers.
    """

    @classmethod
    def setup_anndata(cls, adata, layer=None, spatial_key="spatial", **kwargs):
        fields = [
            LayerField("X", layer, is_count_data=True),
            SpatialCoordsField(obsm_key=spatial_key),
        ]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)


def make_spatial_adata(n: int = 80, n_genes: int = 20) -> AnnData:
    """Create a minimal AnnData with counts layer and spatial coordinates."""
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


@pytest.fixture(scope="module")
def minimal_spatial_adata():
    return make_spatial_adata()


@pytest.fixture(scope="module")
def minimal_model(minimal_spatial_adata):
    _MinimalSpatialModel.setup_anndata(
        minimal_spatial_adata, layer="counts", spatial_key="spatial"
    )
    return _MinimalSpatialModel(minimal_spatial_adata)
