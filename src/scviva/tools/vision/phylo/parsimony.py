"""Fitch-Hartigan parsimony and per-cell plasticity scores.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).

Mirrors R VISION's ``computePlasticityScores`` /
``computeFitchHartiganParsimonyPerNode`` from ``FitchParsimony.R`` and
``AnalysisFunctions.R``.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitch-Hartigan algorithm helpers
# ---------------------------------------------------------------------------


def _build_parent_map(root) -> dict[int, object]:
    """Return {id(clade): parent_clade} for every node in the tree.

    Parameters
    ----------
    root
        Root clade of a ``Bio.Phylo`` tree.

    Returns
    -------
    dict of int to object
        Mapping from ``id(clade)`` to its parent clade (``None`` for the root).
    """
    parent: dict[int, object] = {}

    def _dfs(clade, par) -> None:
        parent[id(clade)] = par
        for child in clade.clades:
            _dfs(child, clade)

    _dfs(root, None)
    return parent


def _fitch_bottom_up(
    root,
    metadata: dict[str, str],
) -> dict[int, frozenset[str]]:
    """Bottom-up Fitch-Hartigan pass for the subtree rooted at *root*.

    For each leaf: possible = {metadata[leaf]}.
    For each internal node: possible = mode of children's possible sets
    (union of most-frequent labels across children).

    Matches R VISION's ``bottomUpFitchHartigan``.

    Parameters
    ----------
    root
        Root clade of the subtree to process.
    metadata : dict of str to str
        Mapping from leaf name to categorical label.

    Returns
    -------
    dict of int to frozenset of str
        Mapping from ``id(node)`` to the set of most-parsimonious labels
        for that node.
    """
    possible: dict[int, frozenset[str]] = {}

    for node in root.find_clades(order="postorder"):
        if node.is_terminal():
            label = metadata.get(node.name)
            possible[id(node)] = frozenset([label] if label is not None else [])
        else:
            counter: Counter = Counter()
            for child in node.clades:
                for label in possible.get(id(child), frozenset()):
                    counter[label] += 1
            if counter:
                max_freq = max(counter.values())
                possible[id(node)] = frozenset(
                    lbl for lbl, freq in counter.items() if freq == max_freq
                )
            else:
                possible[id(node)] = frozenset()

    return possible


def _fitch_top_down(
    root,
    possible: dict[int, frozenset[str]],
) -> dict[int, str | None]:
    """Top-down Fitch-Hartigan pass; returns a maximum-parsimony assignment.

    Tie-breaking is deterministic (lexicographic minimum), unlike R VISION
    which uses ``sample()`` (random).  The parsimony score is identical for
    any valid max-parsimony assignment.

    Matches R VISION's ``topDownFitchHartigan``.

    Parameters
    ----------
    root
        Root clade of the subtree to process.
    possible : dict of int to frozenset of str
        Per-node candidate label sets, as returned by :func:`_fitch_bottom_up`.

    Returns
    -------
    dict of int to str or None
        Mapping from ``id(node)`` to its assigned label.
    """
    assignments: dict[int, str | None] = {}

    def _assign(node, parent_label: str | None) -> None:
        poss = possible.get(id(node), frozenset())
        if not poss:
            assignments[id(node)] = parent_label
        elif parent_label is not None and parent_label in poss:
            assignments[id(node)] = parent_label
        else:
            assignments[id(node)] = min(poss)  # deterministic tie-break
        for child in node.clades:
            _assign(child, assignments[id(node)])

    _assign(root, None)
    return assignments


def _score_parsimony(root, assignments: dict[int, str | None]) -> int:
    """Count label transitions across all edges in the subtree.

    Matches R VISION's ``scoreParsimony``.

    Parameters
    ----------
    root
        Root clade of the subtree to score.
    assignments : dict of int to str or None
        Per-node label assignment, as returned by :func:`_fitch_top_down`.

    Returns
    -------
    int
        Number of edges whose endpoints have different labels.
    """
    parsimony = 0
    for node in root.find_clades():
        par_label = assignments.get(id(node))
        for child in node.clades:
            if par_label != assignments.get(id(child)):
                parsimony += 1
    return parsimony


def _normalized_parsimony(root, metadata: dict[str, str]) -> float:
    """Normalized Fitch-Hartigan parsimony for the subtree rooted at *root*.

    = parsimony / (n_nodes - 1), where n_nodes is the number of nodes
    in the subtree (leaves + internal).

    Matches R VISION's ``computeNormalizedFitchHartiganParsimony(tree,
    metaData, source=node)``.

    Parameters
    ----------
    root
        Root clade of the subtree to score.
    metadata : dict of str to str
        Mapping from leaf name to categorical label.

    Returns
    -------
    float
        Normalized parsimony score in ``[0, 1]``, or 0.0 for single-node
        subtrees.
    """
    possible = _fitch_bottom_up(root, metadata)
    assignments = _fitch_top_down(root, possible)
    parsimony = _score_parsimony(root, assignments)
    n_nodes = sum(1 for _ in root.find_clades())
    return parsimony / (n_nodes - 1) if n_nodes > 1 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_plasticity_scores(
    adata: AnnData,
    newick: str,
) -> None:
    """Compute per-cell Fitch-Hartigan parsimony plasticity scores.

    For every categorical variable in ``adata.obs``, a ``{var}_plasticity``
    column is added to ``adata.obs``.  The score for cell *c* is the average
    normalised Fitch-Hartigan parsimony across all internal nodes on the path
    from the tree root to *c*.

    - **High plasticity**: the metadata label switches often along the
      lineage path → the trait is labile / plastic.
    - **Low plasticity**: the label is conserved along the lineage path.

    Mirrors R VISION's ``computePlasticityScores()``.

    Parameters
    ----------
    adata : AnnData
        Must have ``obs_names`` matching tree leaf labels.
    newick : str
        Newick-format tree string (or the ``adata.uns["vision_tree"]`` value).

    Returns
    -------
    None
        Modifies *adata* in-place, adding one ``{var}_plasticity`` column to
        ``adata.obs`` for each categorical variable found.
    """
    import io

    from Bio import Phylo

    tree = Phylo.read(io.StringIO(newick), "newick")
    terminals = tree.get_terminals()

    name_to_term = {t.name: t for t in terminals}
    parent = _build_parent_map(tree.root)

    # Identify categorical / object columns
    cat_cols: list[str] = [
        col
        for col in adata.obs.columns
        if pd.api.types.is_categorical_dtype(adata.obs[col]) or adata.obs[col].dtype == object
    ]

    if not cat_cols:
        logger.info("No categorical obs columns found; skipping plasticity scores.")
        return

    # Collect all internal nodes (needed for per-node parsimony computation)
    internal_nodes = [c for c in tree.find_clades() if not c.is_terminal()]
    logger.info(
        "Computing plasticity scores for %d categorical variable(s) across %d internal nodes.",
        len(cat_cols),
        len(internal_nodes),
    )

    for col in cat_cols:
        # Build metadata dict: leaf name → label string
        metadata: dict[str, str] = {
            name: str(adata.obs.at[name, col]) for name in adata.obs_names if name in name_to_term
        }
        if not metadata:
            continue

        # Compute normalised parsimony score for each internal node's subtree
        node_score: dict[int, float] = {
            id(node): _normalized_parsimony(node, metadata) for node in internal_nodes
        }

        # Per-leaf plasticity = average node score along root-to-leaf path
        plasticity: dict[str, float] = {}
        for leaf_name in adata.obs_names:
            if leaf_name not in name_to_term:
                plasticity[leaf_name] = np.nan
                continue

            leaf = name_to_term[leaf_name]
            ancestor_scores: list[float] = []

            # Walk from leaf's parent up to (and including) root
            node = leaf
            while parent[id(node)] is not None:
                node = parent[id(node)]
                s = node_score.get(id(node))
                if s is not None:
                    ancestor_scores.append(s)

            plasticity[leaf_name] = float(np.mean(ancestor_scores)) if ancestor_scores else 0.0

        out_col = f"{col}_plasticity"
        adata.obs[out_col] = [plasticity.get(name, np.nan) for name in adata.obs_names]
        logger.info("Plasticity scores stored in adata.obs['%s'].", out_col)
