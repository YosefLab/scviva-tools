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


def _stats_by_name(adata: AnnData, group: str, column: str) -> pd.Series:
    result = adata.uns["rank_genes_groups"]
    return pd.Series(result[column][group], index=result["names"][group])


def test_rank_genes_groups_default_logfoldchanges_are_log2_fold_changes():
    """Log-normalized expression keeps the log2 fold change of the back-transformed means."""
    adata = _make_grouped_adata()
    rank_genes_groups(adata, groupby="group", use_raw=False)

    in_group = (adata.obs["group"] == "g0").to_numpy()
    mean_group = adata.X[in_group].mean(axis=0)
    mean_rest = adata.X[~in_group].mean(axis=0)
    expected = np.log2((np.expm1(mean_group) + 1e-9) / (np.expm1(mean_rest) + 1e-9))

    observed = _stats_by_name(adata, "g0", "logfoldchanges")[adata.var_names]
    np.testing.assert_allclose(observed.to_numpy(), expected, rtol=1e-5)
    assert adata.uns["rank_genes_groups"]["params"]["log_transformed"] is True


def test_rank_genes_groups_not_log_transformed_reports_mean_difference():
    """Z-scored values can be negative: the fold change is NaN, the mean difference is not."""
    adata = _make_grouped_adata()
    rng = np.random.default_rng(1)
    adata.X = rng.normal(size=adata.shape)
    adata.X[adata.obs["group"] == "g0", 1] -= 2.0  # negative group mean, positive rest mean
    adata.X[adata.obs["group"] != "g0", 1] += 2.0

    rank_genes_groups(adata, groupby="group", use_raw=False)
    assert np.isnan(_stats_by_name(adata, "g0", "logfoldchanges")["gene1"])

    rank_genes_groups(adata, groupby="group", use_raw=False, log_transformed=False)
    for group in ["g0", "g1", "g2"]:
        in_group = (adata.obs["group"] == group).to_numpy()
        expected = adata.X[in_group].mean(axis=0) - adata.X[~in_group].mean(axis=0)
        observed = _stats_by_name(adata, group, "logfoldchanges")[adata.var_names]
        assert not observed.isna().any()
        np.testing.assert_allclose(observed.to_numpy(), expected, rtol=1e-5, atol=1e-6)

    # with a reference group, the difference is to the mean of that group
    rank_genes_groups(adata, groupby="group", use_raw=False, reference="g2", log_transformed=False)
    expected = adata.X[(adata.obs["group"] == "g0").to_numpy()].mean(axis=0) - adata.X[
        (adata.obs["group"] == "g2").to_numpy()
    ].mean(axis=0)
    observed = _stats_by_name(adata, "g0", "logfoldchanges")[adata.var_names]
    np.testing.assert_allclose(observed.to_numpy(), expected, rtol=1e-5, atol=1e-6)


def test_wilcoxon_statistics_are_per_gene():
    """Rank sums are taken per gene, so scores and p-values differ between genes.

    Regression test: with pandas 3, ``np.sum`` on the rank DataFrame summed over all cells
    and genes, so every gene of a group got the same score and p-value.
    """
    from scipy.stats import norm, rankdata

    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(2 * n, 5))
    X[:n, 0] += 3.0  # gene0 is higher in group a; continuous values, so no ties
    obs = pd.DataFrame({"group": pd.Categorical(["a"] * n + ["b"] * n)})
    obs.index = [f"cell{i}" for i in range(2 * n)]
    var = pd.DataFrame(index=[f"gene{i}" for i in range(5)])
    adata = AnnData(X=X, obs=obs, var=var)

    # expected values from the rank sum of group a, with VISION's continuity correction
    rank_sum = rankdata(X, axis=0)[:n].sum(axis=0)
    sd = np.sqrt(n * n * (2 * n + 1) / 12.0)
    z = (rank_sum - n * (2 * n + 1) / 2.0) / sd
    expected_score = dict(zip(var.index, rank_sum / (n * n), strict=True))
    expected_pval = dict(zip(var.index, 2 * norm.cdf(-np.abs(z) + 0.5 / sd), strict=True))

    for reference in ["rest", "b"]:  # the two code paths of the Wilcoxon test
        rank_genes_groups(adata, "group", method="wilcoxon", reference=reference)
        res = adata.uns["rank_genes_groups"]
        names = list(res["names"]["a"])
        scores = np.asarray(res["scores"]["a"], dtype=float)
        pvals = np.asarray(res["pvals"]["a"], dtype=float)
        assert names[0] == "gene0"
        assert sorted(names) == list(var.index)
        np.testing.assert_allclose(scores, [expected_score[g] for g in names], rtol=1e-5)
        np.testing.assert_allclose(pvals, [expected_pval[g] for g in names], rtol=1e-6)
        assert len(np.unique(np.round(scores, 6))) == 5
