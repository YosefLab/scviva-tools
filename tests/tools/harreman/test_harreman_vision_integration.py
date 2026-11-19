from __future__ import annotations

import os
import tempfile

import numba
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scviva.tools.harreman._analysis import HarremanAnalysis
from scviva.tools.vision import VisionAnalysis

_NUMBA_CACHE_DIR = os.path.join(tempfile.gettempdir(), "numba_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE_DIR)
numba.config.CACHE_DIR = _NUMBA_CACHE_DIR

SIG_DICT = {
    "SIG_A": [f"gene{i}" for i in range(10)],
    "SIG_B": [f"gene{i}" for i in range(10, 20)],
}

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def adata_with_pca():
    n_obs, n_vars = 120, 30
    rng = np.random.default_rng(42)
    X = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    adata.obsm["X_pca"] = rng.normal(size=(n_obs, 5)).astype(np.float32)
    return adata


@pytest.fixture
def adata_with_prebuilt_weights():
    """Simulates an adata that already went through HarremanAnalysis.setup() —
    i.e. has an obsp["weights"] graph but no obsm embedding VisionAnalysis
    would otherwise pick up (no X_pca).
    """
    n_obs, n_vars = 80, 30
    rng = np.random.default_rng(7)
    X = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)

    w = rng.random((n_obs, n_obs)).astype(np.float32)
    np.fill_diagonal(w, 0)
    w = w / w.sum(axis=1, keepdims=True)
    adata.obsp["weights"] = csr_matrix(w)
    return adata


# ── Tests ────────────────────────────────────────────────────────────────────


def test_scores_only_backward_compatible(adata_with_pca):
    ha = HarremanAnalysis(adata_with_pca)
    ha.vs.load_signatures(dicts=[SIG_DICT])
    ha.vs.analyze_vision(scores_only=True)

    assert "vision_signatures" in adata_with_pca.obsm
    assert adata_with_pca.obsm["vision_signatures"].shape == (adata_with_pca.n_obs, 2)
    # scores_only must not touch the KNN graph or run autocorrelation.
    assert "weights" not in adata_with_pca.obsp
    assert "vision_signature_scores" not in adata_with_pca.uns


def test_full_pipeline_with_explicit_setup_kwargs(adata_with_pca):
    ha = HarremanAnalysis(adata_with_pca)
    ha.vs.load_signatures(dicts=[SIG_DICT])
    ha.vs.analyze_vision(
        setup_kwargs={"compute_neighbors_on_key": "X_pca", "num_neighbors": 15, "exact_knn": True},
        with_differential_expression=True,
    )

    assert "vision_signature_scores" in adata_with_pca.uns
    assert "VISION_Clusters" in adata_with_pca.obs
    # This is the case the stale-_cat_obs_cols bug used to break: with no
    # categorical obs column besides the auto-created VISION_Clusters, the
    # cluster-vs-rest comparison used to be silently dropped.
    assert "vision_signature_differential" in adata_with_pca.uns
    assert "VISION_Clusters" in adata_with_pca.uns["vision_signature_differential"]


def test_full_pipeline_without_differential_expression_skips_de(adata_with_pca):
    ha = HarremanAnalysis(adata_with_pca)
    ha.vs.load_signatures(dicts=[SIG_DICT])
    ha.vs.analyze_vision(
        setup_kwargs={"compute_neighbors_on_key": "X_pca", "num_neighbors": 15, "exact_knn": True},
    )

    assert "vision_signature_scores" in adata_with_pca.uns
    assert "vision_signature_differential" not in adata_with_pca.uns


def test_reuses_existing_weights_graph_instead_of_overwriting(adata_with_prebuilt_weights):
    weights_before = adata_with_prebuilt_weights.obsp["weights"].toarray().copy()

    ha = HarremanAnalysis(adata_with_prebuilt_weights)
    ha.vs.load_signatures(dicts=[SIG_DICT])
    ha.vs.analyze_vision()  # no setup_kwargs -> must reuse obsp["weights"] as-is

    assert "vision_signature_scores" in adata_with_prebuilt_weights.uns
    np.testing.assert_allclose(
        adata_with_prebuilt_weights.obsp["weights"].toarray(),
        weights_before,
        err_msg="pre-existing obsp['weights'] graph was overwritten instead of reused",
    )


def test_reuses_existing_weights_even_when_x_pca_also_present(adata_with_prebuilt_weights):
    """Regression test for the exact Harreman+Vision integration scenario:
    a dataset commonly has BOTH an obsm["X_pca"] embedding (from unrelated
    preprocessing) AND a pre-existing obsp["weights"] graph (built by
    Harreman). Without tracking which tool actually owns the graph, X_pca's
    mere presence used to take priority and silently rebuild (overwrite)
    Harreman's graph with a differently-parameterized one.
    """
    adata_with_prebuilt_weights.obsm["X_pca"] = (
        np.random.default_rng(3)
        .normal(size=(adata_with_prebuilt_weights.n_obs, 5))
        .astype(np.float32)
    )
    weights_before = adata_with_prebuilt_weights.obsp["weights"].toarray().copy()

    ha = HarremanAnalysis(adata_with_prebuilt_weights)
    ha.vs.load_signatures(dicts=[SIG_DICT])
    # no setup_kwargs -> must still reuse obsp["weights"], not rebuild from X_pca
    ha.vs.analyze_vision()

    np.testing.assert_allclose(
        adata_with_prebuilt_weights.obsp["weights"].toarray(),
        weights_before,
        err_msg="existing obsp['weights'] was overwritten because X_pca also happened to exist",
    )


def test_setup_rebuilds_from_x_pca_on_a_second_call_by_the_same_session(adata_with_pca):
    """The weights-reuse fallback must not make a VisionAnalysis session
    "stick" to its own first graph forever: calling setup() again (e.g. with
    different parameters) should still rebuild from X_pca as before, since
    this session -- not some other tool -- owns the existing obsp["weights"].
    """
    va = VisionAnalysis(adata_with_pca)
    va.setup(compute_neighbors_on_key="X_pca", num_neighbors=10, exact_knn=True)
    first_nnz = adata_with_pca.obsp["weights"].nnz

    va.setup(compute_neighbors_on_key="X_pca", num_neighbors=20, exact_knn=True)
    second_nnz = adata_with_pca.obsp["weights"].nnz

    assert second_nnz != first_nnz, "second setup() call did not rebuild the graph"


def test_de_ignores_noncategorical_identifier_obs_column(adata_with_pca):
    """A plain string per-cell identifier column (e.g. a barcode column that
    duplicates obs_names, never cast to a pandas "category" dtype) used to
    crash the one-vs-all comparisons in compute_differential_expression(),
    because _infer_obs_columns() treated any non-numeric column as
    categorical without checking cardinality or dtype.
    """
    adata_with_pca.obs["barcode"] = adata_with_pca.obs_names.to_numpy()  # all-unique, object dtype

    va = VisionAnalysis(adata_with_pca)
    va.load_signatures(dicts=[SIG_DICT])
    va.setup(compute_neighbors_on_key="X_pca", num_neighbors=15, exact_knn=True)
    va.compute_signatures(device="cpu")
    va.compute_differential_expression()  # must not raise

    assert "barcode" not in va._cat_obs_cols
    assert "VISION_Clusters" in va._cat_obs_cols
    assert "vision_signature_differential" in adata_with_pca.uns


def test_compute_signatures_restores_signature_varm_key_after_null_generation(adata_with_pca):
    """compute_signatures() internally scores a transient "random_signatures"
    background set (for the Geary's C permutation null) via a nested call to
    compute_signatures_anndata(), which as a side effect overwrites
    adata.uns["signature_varm_key"] to "random_signatures". It used to leave
    that clobbered value in place, so anything reading
    adata.uns["signature_varm_key"] afterwards (e.g.
    integrate_vision_hotspot_results) would look up the wrong varm key.
    """
    va = VisionAnalysis(adata_with_pca)
    va.load_signatures(dicts=[SIG_DICT])
    va.setup(compute_neighbors_on_key="X_pca", num_neighbors=15, exact_knn=True)
    va.compute_signatures(device="cpu")

    assert adata_with_pca.uns["signature_varm_key"] == "signatures"
    assert "random_signatures" not in adata_with_pca.varm


def test_analyze_vision_requires_loaded_signatures(adata_with_pca):
    ha = HarremanAnalysis(adata_with_pca)
    with pytest.raises(ValueError, match="signatures"):
        ha.vs.analyze_vision(
            setup_kwargs={"compute_neighbors_on_key": "X_pca", "num_neighbors": 15},
        )
