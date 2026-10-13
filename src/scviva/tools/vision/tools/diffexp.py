"""Rank genes for differential expression between groups of cells.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy),
which in turn adapts scanpy's ``rank_genes_groups``
(https://github.com/theislab/scanpy/blob/master/scanpy/tools/_rank_genes_groups.py).
Modified to output AUC for the Wilcoxon test.
"""

from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scanpy import _utils
from scanpy import logging as logg
from scanpy._utils import check_nonnegative_integers
from scipy.sparse import issparse, vstack

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData

_Method = Literal["wilcoxon"] | None
_CorrMethod = Literal["benjamini-hochberg", "bonferroni"]


def _select_top_n(scores: np.ndarray, n_top: int) -> np.ndarray:
    """Return the indices of the *n_top* largest values in *scores*.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of scores to rank.
    n_top : int
        Number of top-scoring indices to return.

    Returns
    -------
    np.ndarray
        Indices into *scores*, sorted from highest to lowest score.
    """
    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]

    return global_indices


def _ranks(X, mask=None, mask_rest=None):
    """Yield per-gene rank chunks of *X*, optionally restricted to two masks.

    Ranks are computed with :meth:`pandas.DataFrame.rank` on gene chunks
    sized to keep peak memory bounded, so that arbitrarily wide expression
    matrices can be ranked without materialising the full dense rank matrix
    at once.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Dense or sparse expression matrix.
    mask : np.ndarray of bool, optional
        Boolean mask selecting cells in the "active" group. When given,
        `mask_rest` must also be given and ranking is restricted to the
        union of the two masks.
    mask_rest : np.ndarray of bool, optional
        Boolean mask selecting cells in the reference group.

    Yields
    ------
    tuple[pd.DataFrame, int, int]
        ``(ranks, left, right)`` where `ranks` holds the per-cell ranks for
        genes ``[left, right)``.
    """
    CONST_MAX_SIZE = 10000000

    n_genes = X.shape[1]

    if issparse(X):
        merge = lambda tpl: vstack(tpl).toarray()  # noqa: E731
        adapt = lambda X: X.toarray()  # noqa: E731
    else:
        merge = np.vstack
        adapt = lambda X: X  # noqa: E731

    masked = mask is not None and mask_rest is not None

    if masked:
        n_cells = np.count_nonzero(mask) + np.count_nonzero(mask_rest)

        def get_chunk(X, left, right):
            return merge((X[mask, left:right], X[mask_rest, left:right]))
    else:
        n_cells = X.shape[0]
        get_chunk = lambda X, left, right: adapt(X[:, left:right])  # noqa: E731

    # Calculate chunk frames
    max_chunk = floor(CONST_MAX_SIZE / n_cells)

    for left in range(0, n_genes, max_chunk):
        right = min(left + max_chunk, n_genes)

        df = pd.DataFrame(data=get_chunk(X, left, right))
        ranks = df.rank()
        yield ranks, left, right


def _tiecorrect(ranks: np.ndarray) -> np.ndarray:
    """Compute the Mann-Whitney tie-correction factor for each column of *ranks*.

    Parameters
    ----------
    ranks : np.ndarray of shape (n_cells, n_genes)
        Per-gene ranks (as produced by :func:`_ranks`).

    Returns
    -------
    np.ndarray of shape (n_genes,)
        Tie-correction factor for each gene.
    """
    size = np.float64(ranks.shape[0])
    if size < 2:
        return np.repeat(1.0, ranks.shape[1])

    arr = np.sort(ranks, axis=0)
    tf = np.insert(arr[1:] != arr[:-1], (0, arr.shape[0] - 1), True, axis=0)
    idx = np.where(tf, np.arange(tf.shape[0])[:, None], 0)
    idx = np.sort(idx, axis=0)
    cnt = np.diff(idx, axis=0).astype(np.float64)

    return 1.0 - (cnt**3 - cnt).sum(axis=0) / (size**3 - size)


def _get_mean_var(X) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-gene mean and (biased) variance of *X*.

    Notes
    -----
    This is a local, independent copy kept separate from
    :func:`~._normalization._get_mean_var` (which uses ``ddof=1``); the two
    are not interchangeable and this one is used only within this module,
    matching the original vendored visionpy implementation.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Dense or sparse expression matrix.

    Returns
    -------
    mean : np.ndarray of shape (n_genes,)
        Per-gene mean.
    var : np.ndarray of shape (n_genes,)
        Per-gene variance (``E[X^2] - E[X]^2``).
    """
    if issparse(X):
        mean = np.array(X.mean(axis=0)).flatten()
        mean_sq = np.array(X.multiply(X).mean(axis=0)).flatten()
    else:
        mean = X.mean(axis=0)
        mean_sq = np.multiply(X, X).mean(axis=0)

    var = mean_sq - np.multiply(mean, mean)

    return mean, var


class _RankGenes:
    """Compute per-group differential-expression statistics.

    Mirrors scanpy's internal ``_RankGenes`` helper, restricted to the
    Wilcoxon rank-sum test (extended to also report the AUC).

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    groups : Iterable[str] or "all"
        Groups to compare, or ``"all"`` for every category in `groupby`.
    groupby : str
        Key in ``adata.obs`` giving the group assignment for each cell.
    reference : str, default "rest"
        Reference group. ``"rest"`` compares each group to the union of
        the others.
    use_raw : bool, default True
        Use ``adata.raw`` if present.
    layer : str or None, default None
        Key in ``adata.layers`` to use instead of ``adata.X`` / ``adata.raw.X``.
    comp_pts : bool, default False
        Also compute the fraction of cells expressing each gene per group.
    """

    def __init__(
        self,
        adata: AnnData,
        groups,
        groupby: str,
        reference: str = "rest",
        use_raw: bool = True,
        layer: str | None = None,
        comp_pts: bool = False,
    ):
        if "log1p" in adata.uns and adata.uns["log1p"]["base"] is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(adata.uns["log1p"]["base"]))
        else:
            self.expm1_func = np.expm1

        self.groups_order, self.groups_masks = _utils.select_groups(adata, groups, groupby)

        # Singlet groups cause division by zero errors
        invalid_groups_selected = set(self.groups_order) & set(
            adata.obs[groupby].value_counts().loc[lambda x: x < 2].index
        )

        if len(invalid_groups_selected) > 0:
            raise ValueError(
                "Could not calculate statistics for groups {} in {} since they only "
                "contain one sample.".format(", ".join(invalid_groups_selected), groupby)
            )

        adata_comp = adata
        if layer is not None:
            if use_raw:
                raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
            X = adata_comp.layers[layer]
        else:
            if use_raw and adata.raw is not None:
                adata_comp = adata.raw
            X = adata_comp.X

        # for correct getnnz calculation
        if issparse(X):
            X.eliminate_zeros()

        self.X = X
        self.var_names = adata_comp.var_names

        self.ireference = None
        if reference != "rest":
            self.ireference = np.where(self.groups_order == reference)[0][0]

        self.means = None
        self.vars = None

        self.means_rest = None
        self.vars_rest = None

        self.comp_pts = comp_pts
        self.pts = None
        self.pts_rest = None

        self.stats = None

        # for logreg only
        self.grouping_mask = adata.obs[groupby].isin(self.groups_order)
        self.grouping = adata.obs.loc[self.grouping_mask, groupby]

    def _basic_stats(self) -> None:
        """Populate per-group (and per-reference) means, variances and pts."""
        n_genes = self.X.shape[1]
        n_groups = self.groups_masks.shape[0]

        self.means = np.zeros((n_groups, n_genes))
        self.vars = np.zeros((n_groups, n_genes))
        self.pts = np.zeros((n_groups, n_genes)) if self.comp_pts else None

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes))
            self.vars_rest = np.zeros((n_groups, n_genes))
            self.pts_rest = np.zeros((n_groups, n_genes)) if self.comp_pts else None
        else:
            mask_rest = self.groups_masks[self.ireference]
            X_rest = self.X[mask_rest]
            self.means[self.ireference], self.vars[self.ireference] = _get_mean_var(X_rest)
            # deleting the next line causes a memory leak for some reason
            del X_rest

        if issparse(self.X):
            get_nonzeros = lambda X: X.getnnz(axis=0)  # noqa: E731
        else:
            get_nonzeros = lambda X: np.count_nonzero(X, axis=0)  # noqa: E731

        for imask, mask in enumerate(self.groups_masks):
            X_mask = self.X[mask]

            if self.comp_pts:
                self.pts[imask] = get_nonzeros(X_mask) / X_mask.shape[0]

            if self.ireference is not None and imask == self.ireference:
                continue

            self.means[imask], self.vars[imask] = _get_mean_var(X_mask)

            if self.ireference is None:
                mask_rest = ~mask
                X_rest = self.X[mask_rest]
                self.means_rest[imask], self.vars_rest[imask] = _get_mean_var(X_rest)
                # this can be costly for sparse data
                if self.comp_pts:
                    self.pts_rest[imask] = get_nonzeros(X_rest) / X_rest.shape[0]
                # deleting the next line causes a memory leak for some reason
                del X_rest

    def wilcoxon(self, tie_correct: bool):
        """Yield ``(group_index, auc, pvals)`` from a Wilcoxon rank-sum test.

        Parameters
        ----------
        tie_correct : bool
            Apply the tie-correction factor to the normal-approximation
            standard deviation.

        Yields
        ------
        tuple[int, np.ndarray, np.ndarray]
            Group index, AUC score per gene, and two-sided p-value per gene.
        """
        from scipy import stats

        self._basic_stats()

        n_genes = self.X.shape[1]
        # First loop: Loop over all genes
        if self.ireference is not None:
            # initialize space for z-scores
            scores = np.zeros(n_genes)
            # initialize space for tie correction coefficients
            if tie_correct:
                T = np.zeros(n_genes)
            else:
                T = 1

            for group_index, mask in enumerate(self.groups_masks):
                if group_index == self.ireference:
                    continue

                mask_rest = self.groups_masks[self.ireference]

                n_active = np.count_nonzero(mask)
                m_active = np.count_nonzero(mask_rest)

                if n_active <= 25 or m_active <= 25:
                    logg.hint(
                        "Few observations in a group for normal approximation "
                        "(<=25). Lower test accuracy."
                    )

                # Calculate rank sums for each chunk for the current mask
                for ranks, left, right in _ranks(self.X, mask, mask_rest):
                    scores[left:right] = np.sum(ranks.iloc[0:n_active, :])
                    if tie_correct:
                        T[left:right] = _tiecorrect(ranks)

                std_dev = np.sqrt(T * n_active * m_active * (n_active + m_active + 1) / 12.0)

                z_scores = (scores - (n_active * ((n_active + m_active + 1) / 2.0))) / std_dev
                z_scores[np.isnan(z_scores)] = 0

                # Continuity correction, matching R VISION's matrix_wilcox /
                # wilcox_subset (Utilities.R:220-224 / Utilities.cpp:203-205):
                # z <- -abs(z); z <- z + .5/sd_u; p <- pnorm(z)*2. Applied
                # after the NaN guard above so already-zeroed (zero-variance)
                # genes keep the neutral p=1 they'd get either way. R's own
                # uncorrected formula can push p slightly over 1 for tiny
                # samples; clip defensively rather than reproducing that.
                with np.errstate(divide="ignore", invalid="ignore"):
                    cc = np.where(std_dev > 0, 0.5 / std_dev, 0.0)
                z_cc = -np.abs(z_scores) + cc
                pvals = np.minimum(2 * stats.distributions.norm.cdf(z_cc), 1.0)

                # auc
                auc = scores / (n_active * m_active)

                yield group_index, auc, pvals
        # If no reference group exists,
        # ranking needs only to be done once (full mask)
        else:
            n_groups = self.groups_masks.shape[0]
            scores = np.zeros((n_groups, n_genes))
            z_scores = np.zeros((n_groups, n_genes))
            n_cells = self.X.shape[0]

            if tie_correct:
                T = np.zeros((n_groups, n_genes))

            for ranks, left, right in _ranks(self.X):
                # sum up adjusted_ranks to calculate W_m,n
                for imask, mask in enumerate(self.groups_masks):
                    scores[imask, left:right] = np.sum(ranks.iloc[mask, :])
                    if tie_correct:
                        T[imask, left:right] = _tiecorrect(ranks)

            for group_index, mask in enumerate(self.groups_masks):
                n_active = np.count_nonzero(mask)

                if tie_correct:
                    T_i = T[group_index]
                else:
                    T_i = 1

                std_dev = np.sqrt(T_i * n_active * (n_cells - n_active) * (n_cells + 1) / 12.0)

                z_scores[group_index, :] = (
                    scores[group_index, :] - (n_active * (n_cells + 1) / 2.0)
                ) / std_dev
                z_scores[group_index, np.isnan(z_scores[group_index, :])] = 0

                # Continuity correction -- see the with-reference branch above.
                with np.errstate(divide="ignore", invalid="ignore"):
                    cc = np.where(std_dev > 0, 0.5 / std_dev, 0.0)
                z_cc = -np.abs(z_scores[group_index, :]) + cc
                pvals = np.minimum(2 * stats.distributions.norm.cdf(z_cc), 1.0)
                # auc
                auc = scores[group_index, :] / (n_active * (n_cells - n_active))

                yield group_index, auc, pvals

    def compute_statistics(
        self,
        method: _Method,
        corr_method: _CorrMethod = "benjamini-hochberg",
        n_genes_user: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        **kwds,
    ) -> None:
        """Run the requested test and assemble the ``self.stats`` table.

        Parameters
        ----------
        method : {"wilcoxon"} or None
            Statistical test to run. Only ``"wilcoxon"`` is implemented;
            ``"t-test"``, ``"t-test_overestim_var"`` and ``"logreg"`` are
            dispatched to methods that are not defined on this class and
            will raise ``AttributeError`` (matching the upstream vendored
            behaviour, where only the Wilcoxon path is implemented).
        corr_method : {"benjamini-hochberg", "bonferroni"}, default "benjamini-hochberg"
            Multiple-testing correction applied to the p-values.
        n_genes_user : int or None, default None
            If given, only the top *n_genes_user* genes per group (by score)
            are retained and a ``"names"`` column is added.
        rankby_abs : bool, default False
            Select top genes by absolute score rather than signed score.
        tie_correct : bool, default False
            Passed through to :meth:`wilcoxon`.
        **kwds
            Passed through to the underlying test method.
        """
        if method in {"t-test", "t-test_overestim_var"}:
            generate_test_results = self.t_test(method)
        elif method == "wilcoxon":
            generate_test_results = self.wilcoxon(tie_correct)
        elif method == "logreg":
            generate_test_results = self.logreg(**kwds)

        self.stats = None

        n_genes = self.X.shape[1]

        for group_index, scores, pvals in generate_test_results:
            group_name = str(self.groups_order[group_index])

            if n_genes_user is not None:
                scores_sort = np.abs(scores) if rankby_abs else scores
                global_indices = _select_top_n(scores_sort, n_genes_user)
                first_col = "names"
            else:
                global_indices = slice(None)
                first_col = "scores"

            if self.stats is None:
                idx = pd.MultiIndex.from_tuples([(group_name, first_col)])
                self.stats = pd.DataFrame(columns=idx)

            if n_genes_user is not None:
                self.stats[group_name, "names"] = self.var_names[global_indices]

            self.stats[group_name, "scores"] = scores[global_indices]

            if pvals is not None:
                self.stats[group_name, "pvals"] = pvals[global_indices]
                if corr_method == "benjamini-hochberg":
                    from statsmodels.stats.multitest import multipletests

                    pvals[np.isnan(pvals)] = 1
                    _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
                elif corr_method == "bonferroni":
                    pvals_adj = np.minimum(pvals * n_genes, 1.0)
                self.stats[group_name, "pvals_adj"] = pvals_adj[global_indices]

            if self.means is not None:
                mean_group = self.means[group_index]
                if self.ireference is None:
                    mean_rest = self.means_rest[group_index]
                else:
                    mean_rest = self.means[self.ireference]
                foldchanges = (self.expm1_func(mean_group) + 1e-9) / (
                    self.expm1_func(mean_rest) + 1e-9
                )  # add small value to remove 0's
                self.stats[group_name, "logfoldchanges"] = np.log2(foldchanges[global_indices])

        if n_genes_user is None:
            self.stats.index = self.var_names


# TODO: Make arguments after groupby keyword only
def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    use_raw: bool = True,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    copy: bool = False,
    method: _Method = "wilcoxon",
    corr_method: _CorrMethod = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
    **kwds,
) -> AnnData | None:
    """
    Rank genes for characterizing groups.

    Expects logarithmized data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    use_raw
        Use `raw` attribute of `adata` if present.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    groups
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups.
    reference
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    n_genes
        The number of genes that appear in the returned tables.
        Defaults to all genes.
    method
        Statistical test. Only `'wilcoxon'` (Wilcoxon rank-sum) is supported;
        `None` also selects it. Any other value raises a :class:`ValueError`.
    corr_method
        p-value correction method.
    tie_correct
        Use tie correction for `'wilcoxon'` scores.
        Used only for `'wilcoxon'`.
    rankby_abs
        Rank genes by the absolute value of the score, not by the
        score. The returned scores are never the absolute values.
    pts
        Compute the fraction of cells expressing the genes.
    key_added
        The key in `adata.uns` information is saved to.
    copy
        If `True`, operate on and return a copy of `adata`; otherwise modify
        `adata` in place and return `None`.
    **kwds
        Accepted for compatibility with scanpy's signature. The legacy
        ``only_positive`` key is honoured (as the inverse of ``rankby_abs``);
        other keys are ignored by the Wilcoxon test.
        For instance, you can pass `penalty='l1'` to try to come up with a
        minimal set of genes that are good predictors (sparse solution meaning
        few non-zero fitted coefficients).

    Returns
    -------
    **names** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    **scores** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group. Ordered according to scores.
    **logfoldchanges** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group. Ordered according to
        scores. Only provided if method is 't-test' like.
        Note: this is an approximation calculated from mean-log values.
    **pvals** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        p-values.
    **pvals_adj** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Corrected p-values.
    **pts** : `pandas.DataFrame` (`.uns['rank_genes_groups']`)
        Fraction of cells expressing the genes for each group.
    **pts_rest** : `pandas.DataFrame` (`.uns['rank_genes_groups']`)
        Only if `reference` is set to `'rest'`.
        Fraction of cells from the union of the rest of each group
        expressing the genes.

    Notes
    -----
    There are slight inconsistencies depending on whether sparse
    or dense data are passed. See `here
    <https://github.com/theislab/scanpy/blob/master/scanpy/tests/test_rank_genes_groups.py>`__.
    """
    if method is None:
        method = "wilcoxon"

    if "only_positive" in kwds:
        rankby_abs = not kwds.pop("only_positive")  # backwards compat

    start = logg.info("ranking genes")
    # Only the Wilcoxon test is ported from scanpy; see `_RankGenes.compute_statistics`.
    avail_methods = {"wilcoxon"}
    if method not in avail_methods:
        raise ValueError(f"Method must be one of {avail_methods}.")

    avail_corr = {"benjamini-hochberg", "bonferroni"}
    if corr_method not in avail_corr:
        raise ValueError(f"Correction method must be one of {avail_corr}.")

    adata = adata.copy() if copy else adata
    _utils.sanitize_anndata(adata)
    # for clarity, rename variable
    if groups == "all":
        groups_order = "all"
    elif isinstance(groups, (str, int)):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = list(groups)
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        if reference != "rest" and reference not in set(groups_order):
            groups_order += [reference]
    if reference != "rest" and reference not in adata.obs[groupby].cat.categories:
        cats = adata.obs[groupby].cat.categories.tolist()
        raise ValueError(f"reference = {reference} needs to be one of groupby = {cats}.")

    if key_added is None:
        key_added = "rank_genes_groups"
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }

    test_obj = _RankGenes(adata, groups_order, groupby, reference, use_raw, layer, pts)

    if check_nonnegative_integers(test_obj.X) and method != "logreg":
        logg.warning(
            "It seems you use rank_genes_groups on the raw count data. "
            "Please logarithmize your data before calling rank_genes_groups."
        )

    # for clarity, rename variable
    n_genes_user = n_genes
    # make sure indices are not OoB in case there are less genes than n_genes
    # defaults to all genes
    if n_genes_user is None or n_genes_user > test_obj.X.shape[1]:
        n_genes_user = test_obj.X.shape[1]

    logg.debug(f"consider {groupby!r} groups:")
    logg.debug(f"with sizes: {np.count_nonzero(test_obj.groups_masks, axis=1)}")

    test_obj.compute_statistics(method, corr_method, n_genes_user, rankby_abs, tie_correct, **kwds)

    if test_obj.pts is not None:
        groups_names = [str(name) for name in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    test_obj.stats.columns = test_obj.stats.columns.swaplevel()

    dtypes = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    for col in test_obj.stats.columns.levels[0]:
        adata.uns[key_added][col] = test_obj.stats[col].to_records(
            index=False, column_dtypes=dtypes[col]
        )

    logg.info(
        "    finished",
        time=start,
        deep=(
            f"added to `.uns[{key_added!r}]`\n"
            "    'names', sorted np.recarray to be indexed by group ids\n"
            "    'scores', sorted np.recarray to be indexed by group ids\n"
            + (
                "    'logfoldchanges', sorted np.recarray to be indexed by group ids\n"
                "    'pvals', sorted np.recarray to be indexed by group ids\n"
                "    'pvals_adj', sorted np.recarray to be indexed by group ids"
                if method in {"t-test", "t-test_overestim_var", "wilcoxon"}
                else ""
            )
        ),
    )
    return adata if copy else None
