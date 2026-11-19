from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from scviva.tools.vision.tools.diffexp import _tiecorrect, rank_genes_groups


def _make_grouped_adata(n_per_group: int = 15, n_genes: int = 4) -> AnnData:
    """Three groups of cells; gene 0 is constant (zero variance) across all cells."""
    rng = np.random.default_rng(0)
    n_groups = 3
    n_obs = n_per_group * n_groups
    X = rng.poisson(3.0, size=(n_obs, n_genes)).astype(float)
    X[:, 0] = 5.0  # constant gene -> zero tie-correction std_dev in the no-reference path
    X = np.log1p(X)  # rank_genes_groups expects logarithmized data
    groups = np.repeat([f"g{i}" for i in range(n_groups)], n_per_group)
    obs = pd.DataFrame({"group": pd.Categorical(groups)}, index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    return AnnData(X=X, obs=obs, var=var)


def test_tiecorrect_small_group_returns_ones_for_every_gene():
    # size < 2 triggers the early-return branch; must be a length-n_genes
    # array of 1.0 (no tie correction), not a length-1 array holding n_genes.
    ranks = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = _tiecorrect(ranks)
    assert result.shape == (ranks.shape[1],)
    np.testing.assert_array_equal(result, np.ones(ranks.shape[1]))


def test_tiecorrect_normal_case_matches_no_ties_expectation():
    # No ties across 5 distinct values per gene -> tie-correction factor is 1.0.
    ranks = np.tile(np.arange(1, 6, dtype=float)[:, None], (1, 3))
    result = _tiecorrect(ranks)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, np.ones(3))


def test_rank_genes_groups_no_reference_path_constant_gene_pvals_adj_not_nan():
    """Regression test for the diffexp.py:399 NaN-mask bug in the no-reference
    (reference='rest', >2 groups) Wilcoxon path with tie_correct=True.

    A constant gene has zero tie-corrected std_dev -> NaN z-score. Before the
    fix this NaN leaked into pvals_adj (bonferroni) undetected because the
    mask checked `scores` (rank sums, never NaN) instead of `z_scores`.
    """
    adata = _make_grouped_adata()
    rank_genes_groups(
        adata,
        groupby="group",
        use_raw=False,
        reference="rest",
        method="wilcoxon",
        corr_method="bonferroni",
        tie_correct=True,
    )
    result = adata.uns["rank_genes_groups"]
    for group_name in result["pvals_adj"].dtype.names:
        pvals_adj = result["pvals_adj"][group_name]
        assert not np.isnan(pvals_adj).any(), (
            f"group {group_name!r} has NaN pvals_adj for a constant gene "
            "(z_scores NaN-mask regression)"
        )


def test_rank_genes_groups_explicit_reference_path_still_handles_constant_gene():
    """The reference-group path (diffexp.py:361) already masked on z_scores
    correctly; keep it covered so a future refactor can't silently regress it.
    """
    adata = _make_grouped_adata()
    rank_genes_groups(
        adata,
        groupby="group",
        use_raw=False,
        reference="g0",
        method="wilcoxon",
        corr_method="bonferroni",
        tie_correct=True,
    )
    result = adata.uns["rank_genes_groups"]
    for group_name in result["pvals_adj"].dtype.names:
        pvals_adj = result["pvals_adj"][group_name]
        assert not np.isnan(pvals_adj).any()


def test_rank_genes_groups_rejects_unimplemented_methods():
    """Only the Wilcoxon test is ported; other scanpy methods must fail clearly."""
    import pytest

    adata = _make_grouped_adata()
    for method in ["t-test", "t-test_overestim_var", "logreg"]:
        with pytest.raises(ValueError, match="Method must be one of"):
            rank_genes_groups(adata, groupby="group", use_raw=False, method=method)


def test_rank_genes_groups_method_none_uses_wilcoxon():
    adata = _make_grouped_adata()
    rank_genes_groups(adata, groupby="group", use_raw=False, method=None)
    assert adata.uns["rank_genes_groups"]["params"]["method"] == "wilcoxon"
