"""VISION: unsupervised transcriptome analysis and signature scoring.

Ported from visionpy (https://github.com/YosefLab/VISION,
https://github.com/yoseflab/visionpy), excluding the interactive web-report
server (Flask app, REST routes, JS/HTML assets). visionpy exposed its
analysis session as a process-global singleton (``AnnDataAccessor``); this
port instead follows :class:`~scviva.tools.harreman.HarremanAnalysis`'s
pattern of an explicit, user-owned session object.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scviva.tools._accessor import adata_delegate
from scviva.tools.vision._constants import (
    CLUSTERS_OBS_KEY,
    META_DIFFERENTIAL_UNS_KEY,
    OBS_DF_SCORES_UNS_KEY,
    PROTEIN_AUTOCORR_UNS_KEY,
    PROTEIN_DIFFERENTIAL_UNS_KEY,
    SIGNATURE_DIFFERENTIAL_UNS_KEY,
    SIGNATURE_SCORES_UNS_KEY,
    SIGNATURES_OBSM_KEY,
    STEP_DE,
    STEP_LATENT,
    STEP_SETUP,
    STEP_SIGNATURES,
    TREE_UNS_KEY,
    VISION_PARAMS_KEY,
    VISION_UNS_KEY,
    WEIGHTS_OBSP_KEY,
)
from scviva.tools.vision._results import VisionResults

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

_STEP_PREREQUISITES: dict[str, str | None] = {
    STEP_SETUP: None,
    STEP_LATENT: None,
    STEP_SIGNATURES: STEP_SETUP,
    STEP_DE: STEP_SETUP,
}


def _pearsonr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between every column of ``a`` (n x p) and every column of ``b`` (n x q)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    norm_a = np.sqrt((a**2).sum(axis=0))
    norm_b = np.sqrt((b**2).sum(axis=0))
    norm_a[norm_a == 0] = 1.0
    norm_b[norm_b == 0] = 1.0
    return (a.T @ b) / np.outer(norm_a, norm_b)


def _covariance_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sample covariance between each column of ``a`` (n x p) and vector ``b`` (n,).

    Matches R VISION's ``evalSigGeneImportance``/``evalSigGeneImportanceSparse``
    (and the compiled ``sigGeneInner``), which use covariance -- not
    correlation -- so that highly-variable genes register as more
    "important", rather than a scale-free [-1, 1] statistic.
    """
    n = a.shape[0]
    a = a - a.mean(axis=0)
    b = b - b.mean()
    return (b @ a) / (n - 1)


def _linkage_to_newick(z: np.ndarray, labels: list) -> str:
    """Convert a scipy linkage matrix to a newick string."""
    n = len(labels)
    nodes = {i: labels[i] for i in range(n)}
    for i, (left, right, dist, _) in enumerate(z):
        left, right = int(left), int(right)
        nodes[n + i] = f"({nodes[left]}:{dist / 2:.4f},{nodes[right]}:{dist / 2:.4f})"
    return nodes[2 * n - 2] + ";"


def _categories(col: pd.Series) -> pd.Index:
    return col.astype("category").cat.categories


class VisionAnalysis:
    """Unsupervised transcriptome analysis and gene-signature scoring.

    Computes a cell neighbor-weight graph, per-cell gene-signature scores,
    their spatial/graph autocorrelation (Geary's C), and one-vs-all
    differential expression of signatures and metadata across clusters —
    mirroring R VISION's ``analyze()`` pipeline.

    Parameters
    ----------
    adata
        Annotated data object.
    norm_data_key
        Key in ``adata.layers`` for scaled/library-size-normalised
        expression. Use ``"use_raw"`` to pull from ``adata.raw``. ``None``
        (default) uses ``adata.X`` directly.

        Matching R VISION's documented contract, this data should be
        **normalized but not log-transformed** -- VISION log-transforms it
        internally wherever needed (:func:`~scviva.tools.vision._utils.log2p1`).
        If it looks already log-transformed (small, non-integer values),
        that internal step is skipped automatically and a warning is logged
        the first time it would otherwise have double-transformed your data;
        pass genuinely non-log-transformed data to avoid relying on that
        detection.
    protein_obsm_key
        Key in ``adata.obsm`` for CITE-seq protein (ADT) data, if any.

    Notes
    -----
    Lower-level VISION functions are available as ``va.tl.<method>``, defaulting to
    this object's AnnData; see :class:`~scviva.tools.vision.VisionToolsAccessor`.

    Examples
    --------
    >>> va = VisionAnalysis(adata)
    >>> va.setup(compute_neighbors_on_key="X_pca")
    >>> va.load_signatures(gmt_files=["h.all.v7.symbols.gmt"])
    >>> va.compute_signatures()
    >>> va.compute_differential_expression()
    >>> results = va.results

    Without a precomputed latent space:

    >>> va = VisionAnalysis(adata)
    >>> va.compute_latent_space()
    >>> va.setup()
    """

    def __init__(
        self,
        adata: AnnData,
        norm_data_key: str | None = None,
        protein_obsm_key: str | None = None,
    ) -> None:
        from anndata import AnnData as _AnnData

        if not isinstance(adata, _AnnData):
            raise TypeError(f"Expected AnnData, got {type(adata).__name__}")

        self._adata = adata
        self._norm_data_key = norm_data_key
        self._protein_obsm_key = protein_obsm_key
        self._signature_varm_key: str | None = None
        self._signature_names_uns_key: str | None = None
        self._completed_steps: set[str] = set()

        # Derived state cached across compute_differential_expression() calls.
        # Declared upfront (unlike visionpy's AnnDataAccessor, which set these
        # dynamically outside __init__) so the object's shape is always clear.
        self._sig_adata: AnnData | None = None
        self._obs_adata: AnnData | None = None
        self._gene_score_sig: dict | None = None
        self._protein_adata: AnnData | None = None
        self._cat_obs_cols, self._numeric_obs_cols = self._infer_obs_columns(adata)

        self._adata.uns.setdefault(VISION_UNS_KEY, {VISION_PARAMS_KEY: {}})

        self.tl = VisionToolsAccessor(self)

    @staticmethod
    def _infer_obs_columns(adata: AnnData) -> tuple[list[str], list[str]]:
        numeric_cols = adata.obs._get_numeric_data().columns.tolist()
        n_obs = adata.n_obs
        cat_cols = [
            c
            for c in adata.obs.columns.tolist()
            if c not in numeric_cols
            # Exclude per-cell identifier columns (e.g. a barcode column
            # duplicating obs_names): every value is unique, so there is no
            # real grouping, and such a column isn't necessarily a pandas
            # "category" dtype, which crashes the one-vs-all comparisons
            # in _compute_one_vs_all_obs_cols/_compute_one_vs_all_signatures.
            and adata.obs[c].nunique(dropna=False) < n_obs
        ]
        return cat_cols, numeric_cols

    # ── Prerequisite checking ────────────────────────────────────────────────

    def _require(self, step: str) -> None:
        prereq = _STEP_PREREQUISITES.get(step)
        if prereq is not None and prereq not in self._completed_steps:
            raise RuntimeError(
                f"{step}() requires {prereq}() to be run first. "
                f"Call va.{prereq}() before continuing."
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def adata(self) -> AnnData:
        """The working AnnData object."""
        return self._adata

    @property
    def norm_data_key(self) -> str | None:
        """Expression layer used for scoring (``None`` uses ``adata.X``)."""
        return self._norm_data_key

    @property
    def var_names(self) -> pd.Index:
        """Gene names, respecting ``norm_data_key == "use_raw"``."""
        return (
            self._adata.raw.var_names
            if self._norm_data_key == "use_raw"
            else self._adata.var_names
        )

    @property
    def results(self) -> VisionResults:
        """Typed view over the analysis results stored in ``adata.uns``/``adata.obsm``."""
        if STEP_SETUP not in self._completed_steps:
            raise RuntimeError("No results available. Run va.setup() first, then analysis steps.")
        return VisionResults.from_adata_uns(self._adata.uns, self._adata.obsm)

    @property
    def is_set_up(self) -> bool:
        """True if :meth:`setup` has been called."""
        return STEP_SETUP in self._completed_steps

    @property
    def has_signatures(self) -> bool:
        """True if signatures have been scored via :meth:`compute_signatures`."""
        return STEP_SIGNATURES in self._completed_steps

    def __repr__(self) -> str:
        steps = sorted(self._completed_steps)
        return f"VisionAnalysis(is_set_up={self.is_set_up}, completed_steps={steps})"

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        compute_neighbors_on_key: str | None = None,
        num_neighbors: int | None = None,
        exact_knn: bool = False,
        tree: str | None = None,
        lca_knn: bool = False,
        lca_min_size: int = 20,
        random_state: int = 0,
    ) -> None:
        """Build the VISION neighbor-weight graph and Louvain cell clusters.

        Parameters
        ----------
        compute_neighbors_on_key
            Key in ``adata.obsm`` to use as the latent space for the KNN
            weight graph. If ``None`` and ``adata.obsm["X_pca"]`` exists,
            that is used automatically. Falls back to L1-normalising
            ``adata.obsp["connectivities"]`` if available.
        num_neighbors
            Number of KNN neighbors for the weight graph. Defaults to
            ``round(sqrt(n_cells))``, matching R VISION.
        exact_knn
            Use exact (brute-force/kd-tree) KNN instead of approximate.
        tree
            Newick string or path to a newick file for PhyloVision mode:
            build KNN weights from phylogenetic tree structure instead of
            an expression/embedding latent space.
        lca_knn
            When ``tree`` is set, use LCA-distance-based tree weights instead
            of cophenetic-distance-based weights.
        lca_min_size
            Minimum clade size for the LCA-based tree KNN weights.
        random_state
            Seed for the Louvain clustering used to build ``VISION_Clusters``
            (``igraph``'s ``community_multilevel`` has no seed parameter of
            its own and is otherwise non-deterministic between runs).
        """
        from scviva.tools.vision.phylo import cluster_cells_tree
        from scviva.tools.vision.tools.knn import (
            compute_knn_weights_anndata,
            compute_knn_weights_from_tree_anndata,
            compute_knn_weights_from_tree_lca_anndata,
        )

        adata = self._adata
        params = adata.uns.setdefault(VISION_UNS_KEY, {}).setdefault(VISION_PARAMS_KEY, {})
        weights_owned_by_vision = params.get("weights_owned_by_vision", False)
        neighbors_key = None
        if tree is not None:
            import os

            newick_str = tree
            if os.path.isfile(tree):
                with open(tree) as fh:
                    newick_str = fh.read().strip()
            if lca_knn:
                compute_knn_weights_from_tree_lca_anndata(adata, newick_str, min_size=lca_min_size)
            else:
                compute_knn_weights_from_tree_anndata(adata, newick_str, K=num_neighbors)
            adata.uns[TREE_UNS_KEY] = newick_str
            cluster_cells_tree(adata, newick_str)
            params["weights_owned_by_vision"] = True
        elif compute_neighbors_on_key is not None:
            compute_knn_weights_anndata(
                adata, obsm_key=compute_neighbors_on_key, K=num_neighbors, exact=exact_knn
            )
            neighbors_key = compute_neighbors_on_key
            params["weights_owned_by_vision"] = True
        elif WEIGHTS_OBSP_KEY in adata.obsp and not weights_owned_by_vision:
            # A weight graph already exists under the same obsp key VISION uses,
            # but VISION itself didn't build it -- i.e. another tool sharing this
            # adata did (e.g. Harreman's own neighbor graph). Reuse it as-is
            # rather than silently overwriting it with a possibly differently
            # -parameterized graph built from an unrelated obsm embedding that
            # merely happens to also be present -- but skip Louvain
            # re-clustering, since we don't know which obsm embedding it came
            # from. Pass compute_neighbors_on_key explicitly to force VISION to
            # build (and own) its own graph instead.
            logger.info(
                "VisionAnalysis: no latent space/tree given; reusing existing "
                "adata.obsp['%s'] weight graph (not built by this VisionAnalysis) as-is.",
                WEIGHTS_OBSP_KEY,
            )
        elif "X_pca" in adata.obsm:
            compute_knn_weights_anndata(adata, obsm_key="X_pca", K=num_neighbors, exact=exact_knn)
            neighbors_key = "X_pca"
            params["weights_owned_by_vision"] = True
        elif "connectivities" in adata.obsp:
            from sklearn.preprocessing import normalize

            adata.obsp[WEIGHTS_OBSP_KEY] = normalize(
                adata.obsp["connectivities"], norm="l1", axis=1
            )
            params["weights_owned_by_vision"] = True
        else:
            raise ValueError(
                "No latent space or neighbor graph found in adata. Call "
                "va.compute_latent_space() first, pass compute_neighbors_on_key, "
                "or pre-compute neighbors with sc.pp.neighbors()."
            )

        if neighbors_key is not None:
            self._cluster_cells(
                obsm_key=neighbors_key,
                num_neighbors=num_neighbors,
                exact_knn=exact_knn,
                random_state=random_state,
            )

        # Invalidate downstream steps so re-running setup with new params
        # doesn't leave stale signature/DE results silently accepted.
        self._completed_steps -= {STEP_SIGNATURES, STEP_DE}
        self._completed_steps.add(STEP_SETUP)
        logger.info("VisionAnalysis: setup complete.")

    def _cluster_cells(
        self,
        obsm_key: str | None = None,
        num_neighbors: int | None = None,
        exact_knn: bool = False,
        random_state: int = 0,
    ) -> None:
        """Louvain clustering matching R VISION's ``clusterCells()``.

        Recomputes a fresh KNN with ``K = min(num_neighbors, 30)`` — separate
        from the ``K = sqrt(n)`` weight matrix used for Geary's C.
        """
        import random

        import igraph as ig

        from scviva.tools.vision.tools.knn import find_knn

        adata = self._adata
        if obsm_key is not None and obsm_key in adata.obsm:
            latent = np.asarray(adata.obsm[obsm_key], dtype=float)
            n = latent.shape[0]
            if num_neighbors is None:
                num_neighbors = max(1, round(np.sqrt(n)))
            k = min(num_neighbors, 30)
            idx, _ = find_knn(latent, k, exact=exact_knn)
            edges = [(i, int(idx[i, j])) for i in range(n) for j in range(k)]
            g = ig.Graph(n=n, edges=edges, directed=True)
        else:
            w = adata.obsp[WEIGHTS_OBSP_KEY].tocoo()
            edges = list(zip(w.row.tolist(), w.col.tolist(), strict=False))
            g = ig.Graph(n=adata.n_obs, edges=edges, directed=True)

        g = g.as_undirected(mode="each")
        # igraph's community_multilevel() has no seed parameter of its own;
        # it draws from whatever generator igraph.set_random_number_generator()
        # last installed (a plain `random.Random` instance by default). Swap
        # in a freshly-seeded instance so this call is reproducible without
        # touching the global `random` module state used elsewhere.
        ig.set_random_number_generator(random.Random(random_state))
        membership = g.community_multilevel().membership
        adata.obs[CLUSTERS_OBS_KEY] = pd.Categorical([str(m) for m in membership])

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def filter_genes(
        self, filters: tuple[str, ...] = ("threshold", "fano"), **kwargs
    ) -> np.ndarray:
        """Compute a boolean gene filter mask; does not mutate ``adata`` in place.

        Parameters
        ----------
        filters
            Names of filters to apply in sequence, e.g. ``"novar"``,
            ``"threshold"``, ``"fano"``. See
            :func:`~scviva.tools.vision.preprocessing.filters.apply_filters`.
        **kwargs
            Forwarded to :func:`~scviva.tools.vision.preprocessing.filters.apply_filters`
            (e.g. ``threshold``, ``num_mad``).

        Returns
        -------
        np.ndarray
            Boolean mask of shape ``(n_vars,)``. Apply with
            ``adata[:, mask].copy()``.
        """
        from scviva.tools.vision.preprocessing.filters import apply_filters

        return apply_filters(self._adata.X, self._adata.var_names, filters=filters, **kwargs)

    def compute_latent_space(
        self, max_components: int = 30, use_permutation_wpca: bool = False, **kwargs
    ) -> None:
        """Compute a PCA latent space, stored as ``adata.obsm["X_pca"]``.

        Parameters
        ----------
        max_components
            Maximum number of principal components to compute.
        use_permutation_wpca
            If ``True``, run permutation weighted PCA and keep only the
            statistically significant components. If ``False``, keep all
            ``max_components`` components (mirrors R VISION's
            ``perm_wPCA=FALSE`` default).
        **kwargs
            Forwarded to :func:`~scviva.tools.vision.compute_latent_space`
            (e.g. ``filters``, ``threshold``, ``num_mad``, ``projection_genes``).
        """
        from scviva.tools.vision.tools.projections import compute_latent_space

        compute_latent_space(
            self._adata,
            norm_data_key=self._norm_data_key if self._norm_data_key != "use_raw" else None,
            max_components=max_components,
            use_permutation_wpca=use_permutation_wpca,
            **kwargs,
        )
        self._completed_steps.add(STEP_LATENT)

    def generate_projections(self, methods: list[str] | None = None, **kwargs) -> None:
        """Generate 2-D visualisation projections (tSNE, UMAP, ISOMap, …).

        Parameters
        ----------
        methods
            Projection methods to compute, chosen from ``"tSNE30"``,
            ``"tSNE10"``, ``"UMAP"``, ``"ISOMap"``, ``"ICA"`` and ``"RBFPCA"``.
            If ``None``, all six methods are computed.
        **kwargs
            Forwarded to :func:`~scviva.tools.vision.generate_projections`
            (e.g. ``obsm_key``, ``K``, ``projection_genes``).
        """
        from scviva.tools.vision.tools.projections import (
            _ALL_PROJECTION_METHODS,
            generate_projections,
        )

        generate_projections(
            self._adata,
            projection_methods=methods if methods is not None else _ALL_PROJECTION_METHODS,
            **kwargs,
        )

    def pool_cells(self, cells_per_partition: int = 10, **kwargs) -> None:
        """Micro-cluster ("pool") cells into supercells for scalable analysis.

        Parameters
        ----------
        cells_per_partition
            Target number of cells per micro-cluster (supercell).
        **kwargs
            Forwarded to :func:`~scviva.tools.vision.apply_micro_clustering`
            (e.g. ``filter_input``, ``filter_threshold``, ``latent_space_key``,
            ``K``, ``uns_key``).
        """
        from scviva.tools.vision.tools.microclusters import apply_micro_clustering

        apply_micro_clustering(self._adata, cells_per_partition=cells_per_partition, **kwargs)

    # ── Signature scoring ────────────────────────────────────────────────────

    def load_signatures(
        self,
        gmt_files: list[str] | None = None,
        dicts: list[dict] | None = None,
        use_raw: bool = False,
        min_signature_genes: int = 5,
        sig_gene_threshold: float = 0.001,
        varm_key: str = "signatures",
        split_signed: bool = True,
    ) -> None:
        """Load gene signatures into ``adata.varm`` ahead of :meth:`compute_signatures`.

        Parameters
        ----------
        gmt_files
            Paths to one or more GMT signature files.
        dicts
            Signature definitions as Python dicts (see
            :func:`~scviva.tools.vision.tools.signature.load_signatures`).
        use_raw
            Use ``adata.raw`` gene names/expression for gene-presence filtering.
        min_signature_genes
            Minimum number of matching genes required to retain a signature.
        sig_gene_threshold
            Minimum fraction of cells expressing a gene for it to be
            eligible for signature membership.
        varm_key
            Key in ``adata.varm`` to store the resulting gene x signature matrix.
        split_signed
            Auto-expand bidirectional (+/-) signatures into ``_UP``/``_DOWN``
            sub-columns after loading, matching R VISION's signature handling.
        """
        from scviva.tools.vision.tools.signature import load_signatures, split_signed_signatures

        load_signatures(
            self._adata,
            gmt_files=gmt_files,
            dicts=dicts,
            use_raw=use_raw,
            min_signature_genes=min_signature_genes,
            sig_gene_threshold=sig_gene_threshold,
            varm_key=varm_key,
        )
        self._signature_varm_key = varm_key
        if split_signed:
            split_signed_signatures(self._adata, varm_key=varm_key, use_raw=use_raw)

    def attach_signatures(self, varm_key: str = "signatures") -> None:
        """Point this session at signatures already loaded into ``adata.varm``.

        Use this instead of :meth:`load_signatures` when signatures were
        already loaded by another call on the same ``adata`` (e.g. a prior
        ``VisionAnalysis`` session, or
        :func:`~scviva.tools.vision.tools.signature.load_signatures` called
        directly) and re-parsing the source GMT/dict would be redundant.

        Parameters
        ----------
        varm_key
            Key in ``adata.varm`` (or ``adata.raw.varm`` when
            ``norm_data_key == "use_raw"``) for the gene x signature weight
            matrix to use.
        """
        store = self._adata.raw.varm if self._norm_data_key == "use_raw" else self._adata.varm
        if varm_key not in store:
            location = "adata.raw.varm" if self._norm_data_key == "use_raw" else "adata.varm"
            raise ValueError(
                f"{location}['{varm_key}'] not found. Call va.load_signatures() "
                "first, or pass the correct varm_key."
            )
        self._signature_varm_key = varm_key

    def compute_signatures(
        self,
        signature_names_uns_key: str | None = None,
        sig_norm_method: str = "znorm_columns",
        device: str = "auto",
        batch_size: int = 1200,
        random_state: int = 0,
    ) -> None:
        """Score loaded signatures per-cell and compute their graph autocorrelation.

        Requires :meth:`setup` (the neighbor-weight graph is used for Geary's C).

        Parameters
        ----------
        signature_names_uns_key
            Key in ``adata.uns`` holding custom signature names, if any.
        sig_norm_method
            Normalisation method for the sparse scoring path. See
            :func:`~scviva.tools.vision.tools.signature.compute_signatures_anndata`.
        device
            Compute device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
            ``"cpu"`` forces the deterministic scipy sparse path (a dense
            torch CPU tensor path is not offered, since its multi-threaded
            matmul is not run-to-run bit-reproducible).
        batch_size
            Number of signatures scored per device batch.
        random_state
            Seed for the Geary's C permutation-null background signature
            generation (see
            :func:`~scviva.tools.vision.tools.signature.generate_permutations_null`).
        """
        self._require(STEP_SIGNATURES)
        if self._signature_varm_key is None:
            raise RuntimeError("No signatures loaded. Call va.load_signatures() first.")

        from scviva.tools.vision.tools.signature import (
            compute_signature_scores,
            compute_signatures_anndata,
        )

        self._signature_names_uns_key = signature_names_uns_key
        compute_signatures_anndata(
            self._adata,
            self._norm_data_key,
            self._signature_varm_key,
            signature_names_uns_key,
            device=device,
            batch_size=batch_size,
        )
        self._adata.uns[SIGNATURE_SCORES_UNS_KEY] = compute_signature_scores(
            self._adata,
            self._norm_data_key,
            self._signature_varm_key,
            sig_norm_method=sig_norm_method,
            random_state=random_state,
        )
        self._completed_steps.add(STEP_SIGNATURES)

    # ── Differential expression ──────────────────────────────────────────────

    def compute_differential_expression(self) -> None:
        """Compute one-vs-all differential expression of metadata and signatures.

        Requires :meth:`setup`. If signatures have been scored via
        :meth:`compute_signatures`, also computes signature differential
        expression, per-signature gene importance, a signature dendrogram,
        and signature clusters — mirroring R VISION's ``ClusterComparisons``.
        """
        self._require(STEP_DE)
        from scviva.tools.vision.tools.signature import compute_obs_df_scores

        adata = self._adata
        # Re-infer categorical/numeric obs columns now, not just at __init__
        # time: setup() may have just added VISION_Clusters (or
        # VISION_Clusters_Tree for PhyloVision) to adata.obs, and the
        # one-vs-all comparisons below should include it as a grouping
        # variable. Without this refresh, datasets with no other categorical
        # obs column would silently get an empty differential-expression
        # result instead of the expected cluster-vs-rest comparison.
        self._cat_obs_cols, self._numeric_obs_cols = self._infer_obs_columns(adata)
        adata.uns[OBS_DF_SCORES_UNS_KEY] = compute_obs_df_scores(adata)
        self._compute_one_vs_all_obs_cols()
        self._persist_meta_differential()

        if self._protein_obsm_key is not None:
            self._compute_protein_autocorrelation()
            self._compute_protein_differential()
            self._persist_protein_differential()

        if SIGNATURES_OBSM_KEY in adata.obsm:
            self._compute_one_vs_all_signatures()
            self._persist_signature_differential()
            self._compute_gene_score_per_signature()
            self._persist_gene_importance()
            self._annotate_latent_components()
            self._compute_signature_dendrogram()
            # Joint FDR must run before signature clustering: R VISION's
            # clusterSignatures() significance gate (FDR < 0.05 & C' > 0.2)
            # reads the *jointly*-corrected FDR (signatures + numeric meta
            # pooled), not each block's independent correction.
            self._recompute_joint_fdr()
            self._compute_signature_clusters()

        self._completed_steps.add(STEP_DE)

    def compute_one_vs_one_de(self, key: str, group1: str, group2: str) -> pd.DataFrame:
        """Wilcoxon one-vs-one differential expression between two groups of ``adata.obs[key]``.

        Full results are also stored in ``adata.uns[f"rank_genes_groups_{key}"]``
        (overwriting any previous comparison for the same ``key``).

        Parameters
        ----------
        key
            Column of ``adata.obs`` defining the groups.
        group1
            Group of ``adata.obs[key]`` to test.
        group2
            Reference group of ``adata.obs[key]`` that ``group1`` is compared
            against.

        Returns
        -------
        :class:`pandas.DataFrame`
            Per-gene results for ``group1`` versus ``group2`` as returned by
            :func:`scanpy.get.rank_genes_groups_df` (gene names, scores,
            log fold changes, p-values and adjusted p-values).
        """
        from scviva.tools.vision.tools.diffexp import rank_genes_groups

        rank_genes_groups(
            self._adata,
            groupby=key,
            groups=[group1],
            reference=group2,
            key_added=f"rank_genes_groups_{key}",
            method="wilcoxon",
            use_raw=self._norm_data_key == "use_raw",
            layer=self._norm_data_key if self._norm_data_key != "use_raw" else None,
        )
        import scanpy as sc

        return sc.get.rank_genes_groups_df(self._adata, group1, key=f"rank_genes_groups_{key}")

    def get_gene_expression(self, genes: str | list[str]) -> np.ndarray:
        """Return the (dense) expression of one or more genes for every cell.

        Expression is read from ``adata.raw`` when ``norm_data_key == "use_raw"``,
        from ``adata.X`` when ``norm_data_key`` is ``None``, and from
        ``adata.layers[norm_data_key]`` otherwise.

        Parameters
        ----------
        genes
            Gene name or list of gene names.

        Returns
        -------
        :class:`numpy.ndarray`
            Dense array of shape ``(n_cells, n_genes)`` (``n_genes == 1`` when a
            single gene name is given).
        """
        from scipy.sparse import issparse

        adata = self._adata
        if self._norm_data_key == "use_raw":
            data = adata.raw[:, genes].X
        elif self._norm_data_key is None:
            data = adata[:, genes].X
        else:
            data = adata[:, genes].layers[self._norm_data_key]
        return data.toarray() if issparse(data) else np.asarray(data)

    def get_genes_by_signature(self, sig_name: str) -> pd.DataFrame:
        """Genes belonging to signature ``sig_name``, indexed by gene name with +/-1 signs.

        Parameters
        ----------
        sig_name
            Name of a signature loaded via :meth:`load_signatures` (or
            :meth:`attach_signatures`) and scored via :meth:`compute_signatures`.

        Returns
        -------
        :class:`pandas.DataFrame`
            Single-column frame indexed by the genes with a non-zero weight in
            the signature, holding each gene's signature weight (sign).
        """
        from scipy.sparse import issparse

        adata = self._adata
        if self._signature_names_uns_key is not None:
            index = np.where(np.asarray(adata.uns[self._signature_names_uns_key]) == sig_name)[0][
                0
            ]
        else:
            index = np.where(np.asarray(adata.obsm[SIGNATURES_OBSM_KEY].columns) == sig_name)[0][0]

        matrix = (
            adata.raw.varm[self._signature_varm_key]
            if self._norm_data_key == "use_raw"
            else adata.varm[self._signature_varm_key]
        )
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.to_numpy()
        matrix = matrix[:, index]
        if issparse(matrix):
            matrix = matrix.toarray().ravel()

        mask = matrix != 0
        return pd.DataFrame(index=self.var_names[mask], data=matrix[mask])

    # ── Private: differential-expression internals (ported from visionpy's
    #    AnnDataAccessor + api.py post-processing helpers) ───────────────────

    def _compute_protein_autocorrelation(self) -> None:
        """Geary's C for CITE-seq protein data, mirroring R VISION's ``fbConsistencyScores``."""
        from scviva.tools.vision.tools.signature import _gearysc_for_dataframe

        adata = self._adata
        mat = adata.obsm[self._protein_obsm_key]
        if isinstance(mat, pd.DataFrame):
            protein_df = mat
        else:
            cols = (
                mat.columns.tolist()
                if hasattr(mat, "columns")
                else [f"Protein_{i}" for i in range(mat.shape[1])]
            )
            protein_df = pd.DataFrame(
                np.asarray(mat, dtype=float), index=adata.obs_names, columns=cols
            )
        weights = adata.obsp[WEIGHTS_OBSP_KEY].tocsr()
        adata.uns[PROTEIN_AUTOCORR_UNS_KEY] = _gearysc_for_dataframe(
            weights, protein_df, compute_pvals=False
        )

    def _compute_protein_differential(self) -> None:
        """One-vs-all Wilcoxon differential for proteins across cluster levels."""
        import anndata as ad

        from scviva.tools.vision.tools.diffexp import rank_genes_groups

        adata = self._adata
        mat = adata.obsm[self._protein_obsm_key]
        if isinstance(mat, pd.DataFrame):
            prot_arr, prot_names = mat.to_numpy(), mat.columns.tolist()
        else:
            prot_arr = np.asarray(mat, dtype=float)
            prot_names = [f"Protein_{i}" for i in range(prot_arr.shape[1])]

        prot_adata = ad.AnnData(prot_arr, obs=adata.obs.loc[:, self._cat_obs_cols].copy())
        prot_adata.var_names = prot_names
        for c in self._cat_obs_cols:
            try:
                rank_genes_groups(
                    prot_adata, groupby=c, key_added=f"rank_genes_groups_{c}", method="wilcoxon"
                )
            except ValueError:
                continue
        self._protein_adata = prot_adata

    def _persist_protein_differential(self) -> None:
        """Persist one-vs-all Wilcoxon protein results to ``adata.uns``.

        Mirrors :meth:`_persist_signature_differential`. Previously this step
        was skipped entirely: the differential was computed into
        ``self._protein_adata`` but never written to ``adata.uns`` -- only
        the protein obsm key string ended up under
        ``PROTEIN_DIFFERENTIAL_UNS_KEY``, so the actual stat/pval results
        were silently discarded. R VISION's ``ClusterComparisons$Proteins``
        is a fully populated stat/pValue/FDR table per group.
        """
        import scanpy as sc

        if self._protein_adata is None:
            return
        result = {}
        for c in self._cat_obs_cols:
            key = f"rank_genes_groups_{c}"
            if key not in self._protein_adata.uns:
                continue
            groups = list(self._protein_adata.obs[c].astype("category").cat.categories)
            frames = []
            for g in groups:
                try:
                    df = sc.get.rank_genes_groups_df(self._protein_adata, group=str(g), key=key)
                    df.insert(0, "group", str(g))
                    frames.append(df)
                except Exception:  # noqa: BLE001
                    continue
            if frames:
                result[c] = pd.concat(frames, ignore_index=True)
        if result:
            self._adata.uns[PROTEIN_DIFFERENTIAL_UNS_KEY] = result

    def _compute_one_vs_all_signatures(self) -> None:
        import anndata as ad

        from scviva.tools.vision.tools.diffexp import rank_genes_groups

        adata = self._adata
        sig_adata = ad.AnnData(adata.obsm[SIGNATURES_OBSM_KEY])
        sig_adata.obs = adata.obs.loc[:, self._cat_obs_cols].copy()
        for c in self._cat_obs_cols:
            rank_genes_groups(
                sig_adata, groupby=c, key_added=f"rank_genes_groups_{c}", method="wilcoxon"
            )
        self._sig_adata = sig_adata

    def _compute_one_vs_all_obs_cols(self) -> None:
        import anndata as ad
        from scipy.stats import chi2_contingency

        from scviva.tools.vision.tools.diffexp import rank_genes_groups

        adata = self._adata
        numeric_data = adata.obs._get_numeric_data().copy()
        n_numeric = numeric_data.shape[1]
        # AnnData requires >= 1 variable; use a zeros placeholder when there
        # is no numeric metadata so categorical chi-squared still runs.
        obs_x = np.log1p(numeric_data.to_numpy()) if n_numeric > 0 else np.zeros((adata.n_obs, 1))
        obs_var_names = list(numeric_data.columns) if n_numeric > 0 else ["_placeholder"]
        obs_adata = ad.AnnData(
            X=obs_x,
            var=pd.DataFrame(index=obs_var_names),
            obs=adata.obs.loc[:, self._cat_obs_cols].copy(),
        )
        for c in list(self._cat_obs_cols):
            try:
                if n_numeric > 0:
                    rank_genes_groups(
                        obs_adata, groupby=c, key_added=f"rank_genes_groups_{c}", method="wilcoxon"
                    )
            except ValueError:
                # A category with a single observation can't be tested.
                self._cat_obs_cols = [c_ for c_ in self._cat_obs_cols if c_ != c]
                continue

            for g in _categories(obs_adata.obs[c]):
                mask = (obs_adata.obs[c] == g).to_numpy()
                obs_pos_masked = obs_adata.obs.iloc[mask]
                obs_neg_masked = obs_adata.obs.iloc[~mask]
                for j in obs_pos_masked.columns:
                    pos_freq = obs_pos_masked[j].value_counts(normalize=False)
                    neg_freq = obs_neg_masked[j].value_counts(normalize=False)
                    freqs = pd.concat([pos_freq, neg_freq], axis=1).fillna(0)
                    grand_total = np.sum(freqs.to_numpy())
                    try:
                        stat, pval, _, _ = chi2_contingency(freqs.to_numpy())
                    except ValueError:
                        # Degenerate contingency table (e.g. a category confined
                        # to a single group) -- mirrors R's matrix_chisq, which
                        # reports "not significant" here, not "maximally
                        # significant".
                        stat, pval = 0.0, 1.0
                    if math.isinf(pval) or math.isnan(pval):
                        pval = 1
                    v = (
                        0.0
                        if (math.isinf(stat) or math.isnan(stat) or grand_total == 0)
                        else np.sqrt(stat / grand_total)
                    )
                    obs_adata.uns[f"chi_sq_{j}_{g}"] = {"stat": v, "pval": pval}

        self._obs_adata = obs_adata

    def _compute_gene_score_per_signature(self) -> None:
        gene_score_sig: dict = {}
        adata = self._adata
        sig_names = (
            adata.uns[self._signature_names_uns_key]
            if self._signature_names_uns_key is not None
            else adata.obsm[SIGNATURES_OBSM_KEY].columns
        )

        for s in sig_names:
            df = self.get_genes_by_signature(s)
            gene_names = df.index
            expr = self.get_gene_expression(gene_names)
            sign = df.to_numpy().ravel()
            sig_scores = adata.obsm[SIGNATURES_OBSM_KEY][s].to_numpy().ravel()
            corrs = _covariance_cols(expr, sig_scores)
            values = (sign * corrs).tolist()
            gene_score_sig[s] = {
                "genes": gene_names.tolist(),
                "values": values,
                "signs": sign.tolist(),
                "geneImportance": dict(zip(gene_names.tolist(), values, strict=False)),
                "sigDict": dict(zip(gene_names.tolist(), sign.tolist(), strict=False)),
            }

        self._gene_score_sig = gene_score_sig

    def _persist_signature_differential(self) -> None:
        """Persist one-vs-all Wilcoxon signature results to ``adata.uns``."""
        import scanpy as sc

        if self._sig_adata is None:
            return
        result = {}
        for c in self._cat_obs_cols:
            key = f"rank_genes_groups_{c}"
            if key not in self._sig_adata.uns:
                continue
            groups = list(self._sig_adata.obs[c].astype("category").cat.categories)
            frames = []
            for g in groups:
                try:
                    df = sc.get.rank_genes_groups_df(self._sig_adata, group=str(g), key=key)
                    df.insert(0, "group", str(g))
                    frames.append(df)
                except Exception:  # noqa: BLE001
                    continue
            if frames:
                result[c] = pd.concat(frames, ignore_index=True)
        if result:
            self._adata.uns[SIGNATURE_DIFFERENTIAL_UNS_KEY] = result

    def _persist_meta_differential(self) -> None:
        """Persist one-vs-all metadata differential results to ``adata.uns``.

        Adds a joint BH-FDR ``"fdr"`` column, pooled per group across both
        the numeric (Wilcoxon) and categorical (chi-squared) blocks --
        mirrors R VISION's ``clusterSigScores()``, which runs one
        ``p.adjust(c(numeric_pvals, factor_pvals), "BH")`` per level rather
        than correcting each block independently. The categorical block
        previously had no FDR correction at all (only a raw ``pval``); the
        numeric block's scanpy-native ``pvals_adj`` (corrected only within
        that group's genes) is left in place alongside the new joint ``fdr``.
        """
        import scanpy as sc
        from statsmodels.stats.multitest import multipletests

        if self._obs_adata is None:
            return
        result = {}
        for c in self._cat_obs_cols:
            col_result = {}
            groups = list(self._obs_adata.obs[c].astype("category").cat.categories)

            key = f"rank_genes_groups_{c}"
            if key in self._obs_adata.uns and self._obs_adata.n_vars > 0:
                frames = []
                for g in groups:
                    try:
                        df = sc.get.rank_genes_groups_df(self._obs_adata, group=str(g), key=key)
                        df.insert(0, "group", str(g))
                        frames.append(df)
                    except Exception:  # noqa: BLE001
                        continue
                if frames:
                    col_result["numeric"] = pd.concat(frames, ignore_index=True)

            cat_rows = []
            for g in groups:
                for j in self._cat_obs_cols:
                    chi_key = f"chi_sq_{j}_{g}"
                    if chi_key in self._obs_adata.uns:
                        info = self._obs_adata.uns[chi_key]
                        cat_rows.append(
                            {
                                "group": str(g),
                                "names": str(j),
                                "cramers_v": info["stat"],
                                "pval": info["pval"],
                            }
                        )
            if cat_rows:
                col_result["categorical"] = pd.DataFrame(cat_rows)

            numeric_df = col_result.get("numeric")
            categorical_df = col_result.get("categorical")
            if numeric_df is not None:
                numeric_df["fdr"] = np.nan
            if categorical_df is not None:
                categorical_df["fdr"] = np.nan
            for g in groups:
                g = str(g)
                num_mask = numeric_df["group"] == g if numeric_df is not None else None
                cat_mask = categorical_df["group"] == g if categorical_df is not None else None
                num_pvals = (
                    numeric_df.loc[num_mask, "pvals"].to_numpy()
                    if numeric_df is not None
                    else np.array([])
                )
                cat_pvals = (
                    categorical_df.loc[cat_mask, "pval"].to_numpy()
                    if categorical_df is not None
                    else np.array([])
                )
                if len(num_pvals) + len(cat_pvals) == 0:
                    continue
                joint_fdr = multipletests(np.concatenate([num_pvals, cat_pvals]), method="fdr_bh")[
                    1
                ]
                if numeric_df is not None:
                    numeric_df.loc[num_mask, "fdr"] = joint_fdr[: len(num_pvals)]
                if categorical_df is not None:
                    categorical_df.loc[cat_mask, "fdr"] = joint_fdr[len(num_pvals) :]

            if col_result:
                result[c] = col_result
        if result:
            self._adata.uns[META_DIFFERENTIAL_UNS_KEY] = result

    def _persist_gene_importance(self) -> None:
        """Persist per-signature gene importance scores to ``adata.uns``."""
        from scviva.tools.vision._constants import GENE_IMPORTANCE_UNS_KEY

        if not self._gene_score_sig:
            return
        result = {}
        for sig_name, info in self._gene_score_sig.items():
            if info["genes"]:
                result[sig_name] = pd.DataFrame(
                    {"importance": info["values"], "sign": info["signs"]}, index=info["genes"]
                )
        if result:
            self._adata.uns[GENE_IMPORTANCE_UNS_KEY] = result

    def _annotate_latent_components(self) -> None:
        """Pearson correlations between PCA components and signature/numeric-obs scores."""
        from scviva.tools.vision._constants import LCA_META_UNS_KEY, LCA_UNS_KEY

        adata = self._adata
        if "X_pca" not in adata.obsm or SIGNATURES_OBSM_KEY not in adata.obsm:
            return
        from scipy.stats import t as t_dist

        latent = np.asarray(adata.obsm["X_pca"], dtype=float)
        n = latent.shape[0]
        pc_labels = [f"PC{i + 1}" for i in range(latent.shape[1])]

        sig_df = adata.obsm[SIGNATURES_OBSM_KEY]
        corr = _pearsonr_matrix(latent, sig_df.to_numpy().astype(float)).T
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(np.maximum(1 - corr**2, 1e-10))
        pvals = 2 * t_dist.sf(np.abs(t_stat), df=n - 2)
        adata.uns[LCA_UNS_KEY] = {
            "sig_labels": sig_df.columns.tolist(),
            "proj_labels": pc_labels,
            "zscores": corr.tolist(),
            "pvals": pvals.tolist(),
        }

        numeric_data = adata.obs._get_numeric_data()
        if numeric_data.shape[1] > 0:
            meta_corr = _pearsonr_matrix(latent, numeric_data.to_numpy().astype(float)).T
            t_meta = meta_corr * np.sqrt(n - 2) / np.sqrt(np.maximum(1 - meta_corr**2, 1e-10))
            adata.uns[LCA_META_UNS_KEY] = {
                "sig_labels": numeric_data.columns.tolist(),
                "proj_labels": pc_labels,
                "zscores": meta_corr.tolist(),
                "pvals": (2 * t_dist.sf(np.abs(t_meta), df=n - 2)).tolist(),
            }

    def _compute_signature_dendrogram(self) -> None:
        """Hierarchically cluster signatures by score vectors; store newick in ``uns``."""
        from scviva.tools.vision._constants import DENDROGRAM_UNS_KEY

        adata = self._adata
        if SIGNATURES_OBSM_KEY not in adata.obsm:
            return
        from scipy.cluster.hierarchy import linkage

        sig_df = adata.obsm[SIGNATURES_OBSM_KEY]
        sig_names = sig_df.columns.tolist()
        if len(sig_names) < 3:
            return
        z = linkage(sig_df.T.to_numpy(), method="ward", metric="euclidean")
        adata.uns[DENDROGRAM_UNS_KEY] = _linkage_to_newick(z, sig_names)

    def _compute_signature_clusters(self) -> None:
        """Cluster signatures via a BIC-selected Gaussian mixture.

        Mirrors R VISION's ``clusterSignatures()``. Only signatures with
        joint FDR < 0.05 *and* Geary's C' > 0.2 are
        clustered; everything else is placed in one extra "non-significant"
        cluster (R: ``compcls[!significant] <- maxcls + 1``). Numeric
        metadata columns meeting the same significance gate are folded into
        the same clustering pass purely to shape the cluster structure (as
        R does, combining ``sigMatrix`` and numeric ``metaData``), but are
        not themselves reported in the output -- only signature -> cluster
        assignments are, matching this key's existing contract.

        R uses ``mclustBIC(..., modelNames="EII")`` -- a Gaussian mixture
        with one *shared* spherical variance across all components.
        scikit-learn has no exact "tied + spherical" option, so this uses
        ``GaussianMixture(covariance_type="spherical")`` (one spherical
        variance *per* component) as the closest available approximation.
        """
        from sklearn.mixture import GaussianMixture

        from scviva.tools.vision._constants import SIG_CLUSTERS_UNS_KEY

        adata = self._adata
        if SIGNATURES_OBSM_KEY not in adata.obsm:
            return

        sig_df = adata.obsm[SIGNATURES_OBSM_KEY]
        sig_names = sig_df.columns.tolist()
        n_sigs = len(sig_names)
        if n_sigs < 3:
            adata.uns[SIG_CLUSTERS_UNS_KEY] = dict.fromkeys(sig_names, 1)
            return

        def _is_significant(name: str, table) -> bool:
            if table is None or name not in table.index:
                return False
            row = table.loc[name]
            return bool(row["fdr"] < 0.05 and row["c_prime"] > 0.2)

        sig_scores_table = adata.uns.get(SIGNATURE_SCORES_UNS_KEY)
        obs_scores_table = adata.uns.get(OBS_DF_SCORES_UNS_KEY)

        numeric_data = (
            adata.obs[self._numeric_obs_cols]
            if self._numeric_obs_cols
            else pd.DataFrame(index=adata.obs_names)
        )
        numeric_data = numeric_data.loc[:, numeric_data.notna().all(axis=0)]

        # variable name -> (per-cell score vector, is a real signature?)
        candidate_vars: dict[str, tuple[np.ndarray, bool]] = {
            s: (sig_df[s].to_numpy(dtype=float), True) for s in sig_names
        }
        for col in numeric_data.columns:
            candidate_vars[col] = (numeric_data[col].to_numpy(dtype=float), False)

        significant_vars = [
            v
            for v, (_, is_sig_col) in candidate_vars.items()
            if _is_significant(v, sig_scores_table if is_sig_col else obs_scores_table)
        ]

        if len(significant_vars) > 5:
            mat = np.column_stack([candidate_vars[v][0] for v in significant_vars])
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds == 0] = 1.0
            mat_z = (mat - means) / stds

            n_cells = mat_z.shape[0]
            if n_cells > 5000:
                rng = np.random.default_rng(0)
                cell_idx = rng.choice(n_cells, 5000, replace=False)
                mat_z = mat_z[cell_idx]

            points = mat_z.T  # (n_vars, n_cells_sub) -- one "sample" per variable

            best_bic = np.inf
            best_labels = None
            max_components = min(15, points.shape[0])
            for k in range(1, max_components + 1):
                try:
                    gmm = GaussianMixture(
                        n_components=k, covariance_type="spherical", random_state=0, n_init=1
                    ).fit(points)
                except ValueError:
                    continue
                bic = gmm.bic(points)
                if bic < best_bic:
                    best_bic = bic
                    best_labels = gmm.predict(points)

            if best_labels is None:
                labels_by_var = dict.fromkeys(significant_vars, 1)
                max_label = 1
            else:
                labels_by_var = {
                    v: int(lbl) + 1 for v, lbl in zip(significant_vars, best_labels, strict=False)
                }
                max_label = max(labels_by_var.values())
        else:
            labels_by_var = dict.fromkeys(significant_vars, 1)
            max_label = 1

        junk_label = max_label + 1
        adata.uns[SIG_CLUSTERS_UNS_KEY] = {s: labels_by_var.get(s, junk_label) for s in sig_names}

    def _recompute_joint_fdr(self) -> None:
        """BH-correct p-values for signatures and metadata jointly.

        R VISION applies ``p.adjust(method="BH")`` over all tests (signatures
        + numerical meta) in a single call. Reproducing that here prevents
        the FDR denominator from differing between the two tools.
        """
        from statsmodels.stats.multitest import multipletests

        adata = self._adata
        obs_df = adata.uns.get(OBS_DF_SCORES_UNS_KEY)
        sig_df = adata.uns.get(SIGNATURE_SCORES_UNS_KEY)
        if obs_df is None or sig_df is None:
            return

        pvals_obs = obs_df["pvals"].values
        pvals_sig = sig_df["pvals"].values
        all_fdr = multipletests(np.concatenate([pvals_obs, pvals_sig]), method="fdr_bh")[1]

        n_obs = len(pvals_obs)
        obs_df, sig_df = obs_df.copy(), sig_df.copy()
        obs_df["fdr"] = all_fdr[:n_obs]
        sig_df["fdr"] = all_fdr[n_obs:]
        adata.uns[OBS_DF_SCORES_UNS_KEY] = obs_df
        adata.uns[SIGNATURE_SCORES_UNS_KEY] = sig_df


# ── Accessor classes ──────────────────────────────────────────────────────────


class VisionToolsAccessor:
    """Lower-level VISION tool functions accessible as ``va.tl.<method>``.

    Created automatically by :class:`~scviva.tools.vision.VisionAnalysis`;
    users do not instantiate it directly.

    Parameters
    ----------
    va
        The analysis object whose AnnData is used when ``adata`` is not passed.
    """

    def __init__(self, va: VisionAnalysis) -> None:
        self._va = va

    def _resolve(self, adata: AnnData | None) -> AnnData:
        return adata if adata is not None else self._va._adata

    compute_knn_weights = adata_delegate(
        "scviva.tools.vision.tools.knn", "compute_knn_weights_anndata"
    )
    compute_plasticity_scores = adata_delegate(
        "scviva.tools.vision.phylo", "compute_plasticity_scores"
    )
    cluster_cells_tree = adata_delegate("scviva.tools.vision.phylo", "cluster_cells_tree")
    apply_micro_clustering = adata_delegate(
        "scviva.tools.vision.tools.microclusters", "apply_micro_clustering"
    )
    pool_matrix = adata_delegate("scviva.tools.vision.tools.microclusters", "pool_matrix_anndata")
    pool_metadata = adata_delegate(
        "scviva.tools.vision.tools.microclusters", "pool_metadata_anndata"
    )
    generate_projections = adata_delegate(
        "scviva.tools.vision.tools.projections", "generate_projections"
    )

    def rank_genes_groups(self, adata: AnnData | None = None, **kwargs):
        """Rank genes for characterizing groups.

        Forwards to :func:`~scviva.tools.vision.rank_genes_groups` (imported lazily
        because it depends on scanpy). ``adata`` defaults to the AnnData held by the
        analysis object.

        Parameters
        ----------
        adata
            AnnData to use. Defaults to the analysis object's AnnData.
        **kwargs
            Keyword arguments forwarded to :func:`~scviva.tools.vision.rank_genes_groups`
            (e.g. ``groupby``, ``method``, ``reference``, ``n_genes``).
        """
        from scviva.tools.vision.tools.diffexp import rank_genes_groups

        return rank_genes_groups(self._resolve(adata), **kwargs)
