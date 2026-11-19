from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

from scviva.tools.harreman._analysis import HarremanAnalysis
from scviva.tools.vision import phylo as vision_phylo
from scviva.tools.vision._analysis import VisionAnalysis
from scviva.tools.vision._constants import (
    CLUSTERS_OBS_KEY,
    META_DIFFERENTIAL_UNS_KEY,
    SIGNATURE_SCORES_UNS_KEY,
    STEP_SETUP,
    STEP_SIGNATURES,
)
from scviva.tools.vision._results import VisionResults
from scviva.tools.vision.preprocessing.normalization import get_normalized_copy
from scviva.tools.vision.tools import diffexp as vision_diffexp
from scviva.tools.vision.tools import knn as vision_knn
from scviva.tools.vision.tools import microclusters as vision_microclusters
from scviva.tools.vision.tools import projections as vision_projections
from scviva.tools.vision.tools import signature as vision_signature


def _make_adata(n_obs: int = 8, n_vars: int = 6, *, obs: pd.DataFrame | None = None) -> AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(n_obs, n_vars)).astype(float)
    if obs is None:
        obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    return AnnData(X=X, obs=obs, var=var)


def _row_normalized_complete_graph(n_obs: int) -> csr_matrix:
    weights = np.ones((n_obs, n_obs), dtype=float)
    np.fill_diagonal(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return csr_matrix(weights)


def _signature_matrix(adata: AnnData, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = ["sig0", "sig1"]
    data = np.zeros((adata.n_vars, len(columns)), dtype=float)
    for i, _ in enumerate(columns):
        data[i % adata.n_vars, i] = 1.0
        data[(i + 1) % adata.n_vars, i] = 1.0
    return pd.DataFrame(data, index=adata.var_names, columns=columns)


def test_compute_signature_scores_restores_signature_varm_key(monkeypatch):
    adata = _make_adata()
    adata.obsp["weights"] = _row_normalized_complete_graph(adata.n_obs)
    adata.varm["signatures"] = _signature_matrix(adata)
    adata.obsm["vision_signatures"] = pd.DataFrame(
        np.arange(adata.n_obs * 2).reshape(adata.n_obs, 2),
        index=adata.obs_names,
        columns=["sig0", "sig1"],
    )
    adata.uns["norm_data_key"] = None
    adata.uns["signature_varm_key"] = "signatures"

    def fake_gearys_c(_weights, ranked):
        return np.zeros(ranked.shape[0], dtype=float)

    def fake_generate_null(_adata, _norm_data_key, _signature_varm_key, random_state=0):
        random_sig = pd.DataFrame(
            {"RANDOM_BG_0_0": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
            index=_adata.var_names,
        )
        clusters = pd.Series([0, 0], index=["sig0", "sig1"])
        random_clusters = pd.Series([0], index=["RANDOM_BG_0_0"])
        return random_sig, clusters, random_clusters

    monkeypatch.setattr(vision_signature, "_gearys_c", fake_gearys_c)
    monkeypatch.setattr(vision_signature, "generate_permutations_null", fake_generate_null)

    vision_signature.compute_signature_scores(adata, None, "signatures")

    assert adata.uns["signature_varm_key"] == "signatures"
    assert "random_signatures" not in adata.varm
    assert list(adata.obsm["vision_signatures"].columns) == ["sig0", "sig1"]


def test_setup_reuses_existing_weights_when_pca_is_present():
    adata = _make_adata()
    rng = np.random.default_rng(1)
    adata.obsm["X_pca"] = rng.normal(size=(adata.n_obs, 3))
    original_weights = _row_normalized_complete_graph(adata.n_obs)
    adata.obsp["weights"] = original_weights.copy()

    VisionAnalysis(adata).setup(exact_knn=True)

    np.testing.assert_allclose(adata.obsp["weights"].toarray(), original_weights.toarray())


def test_infer_obs_columns_ignores_plain_identifier_strings():
    obs = pd.DataFrame(
        {
            "barcode": [f"AAAC{i}" for i in range(6)],
            "condition": pd.Categorical(["ctrl", "stim", "ctrl", "stim", "ctrl", "stim"]),
            "score": np.arange(6, dtype=float),
        },
        index=[f"cell{i}" for i in range(6)],
    )
    adata = _make_adata(n_obs=6, obs=obs)

    categorical, numeric = VisionAnalysis._infer_obs_columns(adata)

    assert categorical == ["condition"]
    assert numeric == ["score"]


def test_compute_de_uses_obs_columns_added_after_init(monkeypatch):
    adata = _make_adata()
    va = VisionAnalysis(adata)
    adata.obs[CLUSTERS_OBS_KEY] = pd.Categorical(["0", "0", "0", "0", "1", "1", "1", "1"])
    va._completed_steps.add(STEP_SETUP)

    def fake_obs_scores(_adata):
        return pd.DataFrame(
            {"c_prime": [0.0], "pvals": [1.0], "fdr": [1.0]},
            index=[CLUSTERS_OBS_KEY],
        )

    monkeypatch.setattr(vision_signature, "compute_obs_df_scores", fake_obs_scores)

    va.compute_differential_expression()

    assert CLUSTERS_OBS_KEY in adata.uns[META_DIFFERENTIAL_UNS_KEY]


def test_harreman_compute_vision_signatures_uses_default_signature_keys(monkeypatch):
    adata = _make_adata()
    calls = []

    def fake_compute(adata_arg, norm_data_key, signature_varm_key, signature_names_uns_key=None):
        calls.append((adata_arg, norm_data_key, signature_varm_key, signature_names_uns_key))
        return pd.DataFrame(index=adata_arg.obs_names)

    monkeypatch.setattr(vision_signature, "compute_signatures_anndata", fake_compute)

    HarremanAnalysis(adata).vs.compute_vision_signatures()

    assert calls == [(adata, None, "signatures", None)]


def test_harreman_analyze_vision_runs_full_signature_pipeline_without_de(monkeypatch):
    adata = _make_adata()
    adata.obsp["weights"] = _row_normalized_complete_graph(adata.n_obs)
    adata.varm["signatures"] = _signature_matrix(adata)

    def fake_signature_scores(
        _adata,
        _norm_data_key,
        _signature_varm_key,
        sig_norm_method="znorm_columns",
        random_state=0,
    ):
        return pd.DataFrame(
            {"c_prime": [0.0, 0.0], "pvals": [1.0, 1.0], "fdr": [1.0, 1.0]},
            index=["sig0", "sig1"],
        )

    monkeypatch.setattr(vision_signature, "compute_signature_scores", fake_signature_scores)

    HarremanAnalysis(adata).vs.analyze_vision(scores_only=False, signature_varm_key="signatures")

    assert "vision_signatures" in adata.obsm
    assert SIGNATURE_SCORES_UNS_KEY in adata.uns
    assert META_DIFFERENTIAL_UNS_KEY not in adata.uns


def test_vision_analysis_load_signatures_honors_custom_varm_key():
    adata = _make_adata()

    VisionAnalysis(adata).load_signatures(
        dicts=[{"custom_sig": ["gene0", "gene1"]}],
        min_signature_genes=1,
        varm_key="custom_signatures",
    )

    assert "custom_signatures" in adata.varm
    assert list(adata.varm["custom_signatures"].columns) == ["custom_sig"]


def test_load_signatures_use_raw_stores_raw_aligned_signatures():
    adata = _make_adata(n_vars=3)
    raw = AnnData(
        X=np.ones((adata.n_obs, 5)),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=["gene0", "gene1", "gene2", "gene3", "gene4"]),
    )
    adata.raw = raw

    vision_signature.load_signatures(
        adata,
        use_raw=True,
        dicts=[{"raw_sig": ["gene0", "gene1", "gene2", "gene3"]}],
        min_signature_genes=1,
    )

    assert "signatures" in adata.raw.varm
    assert list(adata.raw.varm["signatures"].index) == adata.raw.var_names.tolist()


def test_generate_permutations_null_matches_small_signature_clusters():
    adata = _make_adata(n_vars=6)
    adata.varm["signatures"] = _signature_matrix(adata, columns=["sig0"])

    _, clusters, random_clusters = vision_signature.generate_permutations_null(
        adata, None, "signatures"
    )

    for cluster in clusters:
        assert (random_clusters == cluster).any()


def test_compute_obs_df_scores_ignores_constant_numeric_column():
    adata = _make_adata(n_obs=10)
    adata.obs["in_tissue"] = 1  # constant, e.g. every spot passed a QC flag
    adata.obs["score"] = np.arange(10, dtype=float)
    adata.obsp["weights"] = _row_normalized_complete_graph(adata.n_obs)

    result = vision_signature.compute_obs_df_scores(adata)

    assert result.loc["in_tissue", "pvals"] == 1.0
    assert result.loc["in_tissue", "c_prime"] == 0.0
    assert np.isfinite(result.loc["score", "c_prime"])


def test_has_signatures_reflects_completed_steps():
    va = VisionAnalysis(_make_adata())

    assert va.has_signatures is False

    va._completed_steps.add(STEP_SIGNATURES)

    assert va.has_signatures is True


def test_norm_data_key_property_returns_constructor_value():
    adata = _make_adata(n_vars=4)
    adata.layers["mylayer"] = adata.X.copy()

    va = VisionAnalysis(adata, norm_data_key="mylayer")

    assert va.norm_data_key == "mylayer"
    assert VisionAnalysis(adata).norm_data_key is None


def test_results_property_requires_setup_first():
    adata = _make_adata()
    va = VisionAnalysis(adata)

    with pytest.raises(RuntimeError, match="setup"):
        _ = va.results

    va._completed_steps.add(STEP_SETUP)

    results = va.results

    assert isinstance(results, VisionResults)
    assert results.signature_scores is None


def test_repr_reflects_setup_state():
    va = VisionAnalysis(_make_adata())

    assert repr(va) == "VisionAnalysis(is_set_up=False, completed_steps=[])"

    va._completed_steps.add(STEP_SETUP)

    assert repr(va) == f"VisionAnalysis(is_set_up=True, completed_steps=['{STEP_SETUP}'])"


def test_get_normalized_copy_znorm_columns_produces_row_zscores():
    rng = np.random.default_rng(0)
    X = rng.poisson(5.0, size=(4, 10)).astype(float)

    normalized = get_normalized_copy(X, method="znorm_columns")

    assert normalized.shape == X.shape
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-8)
    np.testing.assert_allclose(normalized.std(axis=1, ddof=1), 1.0, atol=1e-8)


def test_compute_one_vs_one_de_returns_wilcoxon_results_between_two_groups():
    adata = _make_adata(n_obs=8, n_vars=6)
    adata.obs["grp"] = pd.Categorical(["A", "A", "A", "A", "B", "B", "B", "B"])

    result = VisionAnalysis(adata).compute_one_vs_one_de("grp", "A", "B")

    assert isinstance(result, pd.DataFrame)
    assert "names" in result.columns
    assert set(result["names"]) == set(adata.var_names)


def test_vision_tl_accessor_delegates_to_underlying_functions(monkeypatch):
    adata = _make_adata()
    va = VisionAnalysis(adata)
    calls: dict[str, tuple] = {}

    def _recorder(name):
        def _fake(adata_arg, **kwargs):
            calls[name] = (adata_arg, kwargs)
            return f"{name}-result"

        return _fake

    monkeypatch.setattr(
        vision_knn, "compute_knn_weights_anndata", _recorder("compute_knn_weights")
    )
    monkeypatch.setattr(
        vision_phylo, "compute_plasticity_scores", _recorder("compute_plasticity_scores")
    )
    monkeypatch.setattr(vision_phylo, "cluster_cells_tree", _recorder("cluster_cells_tree"))
    monkeypatch.setattr(
        vision_microclusters, "apply_micro_clustering", _recorder("apply_micro_clustering")
    )
    monkeypatch.setattr(vision_microclusters, "pool_matrix_anndata", _recorder("pool_matrix"))
    monkeypatch.setattr(vision_microclusters, "pool_metadata_anndata", _recorder("pool_metadata"))
    monkeypatch.setattr(
        vision_projections, "generate_projections", _recorder("generate_projections")
    )
    monkeypatch.setattr(vision_diffexp, "rank_genes_groups", _recorder("rank_genes_groups"))

    va.tl.compute_knn_weights(K=5)
    va.tl.compute_plasticity_scores(tree="newick")
    va.tl.cluster_cells_tree(tree="newick")
    va.tl.apply_micro_clustering(cells_per_partition=10)
    va.tl.pool_matrix()
    va.tl.pool_metadata()
    va.tl.generate_projections(methods=["tSNE30"])
    va.tl.rank_genes_groups(groupby="condition")

    assert calls["compute_knn_weights"] == (adata, {"K": 5})
    assert calls["compute_plasticity_scores"] == (adata, {"tree": "newick"})
    assert calls["cluster_cells_tree"] == (adata, {"tree": "newick"})
    assert calls["apply_micro_clustering"] == (adata, {"cells_per_partition": 10})
    assert calls["pool_matrix"] == (adata, {})
    assert calls["pool_metadata"] == (adata, {})
    assert calls["generate_projections"] == (adata, {"methods": ["tSNE30"]})
    assert calls["rank_genes_groups"] == (adata, {"groupby": "condition"})

    # An explicit adata argument should override the session's own adata.
    other_adata = _make_adata()
    va.tl.compute_knn_weights(other_adata, K=3)
    assert calls["compute_knn_weights"] == (other_adata, {"K": 3})
