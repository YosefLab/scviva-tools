"""Gene-membership checks in the Harreman interaction-database builders.

Under pandas 3, ``AnnData.var_names`` is an Arrow-backed string index. ``Series.isin``
against it re-converts the whole index on every call, so the per-interaction loops in
``extract_lr_pairs`` / ``build_LR_matrix`` took 25-50 minutes on real Visium data
(~31k genes, ~2k CellChatDB interactions). These tests pin both the results and a
time budget on an Arrow-string gene index of realistic size.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scviva.tools.harreman.preprocessing.database as database_mod

pooch = pytest.importorskip("pooch")

N_GENES = 30_000
N_INTERACTIONS = 400
TIME_BUDGET_S = 5.0


def _arrow_gene_index(n: int) -> pd.Index:
    return pd.Index([f"Gene{i}" for i in range(n)], dtype="str")


def _patch_cellchat(monkeypatch, interaction: pd.DataFrame, complex_df: pd.DataFrame):
    monkeypatch.setattr(pooch, "retrieve", lambda **kw: f"/tmp/{kw['fname']}")
    monkeypatch.setattr(pooch, "os_cache", lambda name: f"/cache/{name}")

    def read_csv(path, index_col=0):
        return interaction.copy() if "interaction" in path else complex_df.copy()

    monkeypatch.setattr(database_mod.pd, "read_csv", read_csv)


def test_extract_lr_pairs_keeps_only_genes_present_in_var_names(monkeypatch):
    interaction = pd.DataFrame(
        {
            "ligand": ["GeneA", "CplxL", "Missing"],
            "receptor": ["CplxR", "GeneB", "GeneA"],
            "annotation": ["a", "b", "c"],
        },
        index=["p0", "p1", "p2"],
    )
    complex_df = pd.DataFrame(
        {"subunit_1": ["GeneC", "GeneD"], "subunit_2": ["Absent", "GeneE"]},
        index=["CplxL", "CplxR"],
    )
    _patch_cellchat(monkeypatch, interaction, complex_df)
    var = pd.Index(["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"], dtype="str")
    adata = AnnData(X=np.zeros((1, len(var))), var=pd.DataFrame(index=var))

    database_mod.extract_lr_pairs(adata, species="mouse")

    lig = adata.uns["ligand"].apply(lambda r: [g for g in r if pd.notna(g)], axis=1)
    rec = adata.uns["receptor"].apply(lambda r: [g for g in r if pd.notna(g)], axis=1)
    assert lig.to_dict() == {"p0": ["GeneA"], "p1": ["GeneC"], "p2": []}
    assert rec.to_dict() == {"p0": ["GeneD", "GeneE"], "p1": ["GeneB"], "p2": ["GeneA"]}


def test_extract_lr_pairs_is_fast_on_arrow_string_var_names(monkeypatch):
    rng = np.random.default_rng(0)
    picks = rng.integers(0, N_GENES * 2, size=(N_INTERACTIONS, 2))  # ~half absent
    interaction = pd.DataFrame(
        {
            "ligand": [f"Gene{i}" for i in picks[:, 0]],
            "receptor": [f"Gene{i}" for i in picks[:, 1]],
            "annotation": ["secreted"] * N_INTERACTIONS,
        },
        index=[f"p{i}" for i in range(N_INTERACTIONS)],
    )
    _patch_cellchat(monkeypatch, interaction, pd.DataFrame(index=pd.Index([], dtype="str")))
    var = _arrow_gene_index(N_GENES)
    adata = AnnData(X=np.zeros((1, N_GENES), dtype=np.float32), var=pd.DataFrame(index=var))

    start = time.perf_counter()
    database_mod.extract_lr_pairs(adata, species="human")
    elapsed = time.perf_counter() - start

    assert elapsed < TIME_BUDGET_S, f"extract_lr_pairs took {elapsed:.1f}s"


def test_build_lr_matrix_matches_expected_and_is_fast():
    index = _arrow_gene_index(N_GENES)
    cols = [f"p{i}" for i in range(N_INTERACTIONS)]
    database = pd.DataFrame(index=cols)
    ligands = pd.DataFrame(
        {"Ligand0": [f"GENE{i}" for i in range(N_INTERACTIONS)]}, index=cols
    )  # upper case on purpose: matching is case-insensitive
    receptors = pd.DataFrame(
        {
            "Receptor0": [f"Gene{i + 1}" for i in range(N_INTERACTIONS)],
            "Receptor1": [None] * (N_INTERACTIONS - 1) + ["NotAGene"],
        },
        index=cols,
    )

    start = time.perf_counter()
    matrix = database_mod.build_LR_matrix(index, database, ligands, receptors)
    elapsed = time.perf_counter() - start

    assert elapsed < TIME_BUDGET_S, f"build_LR_matrix took {elapsed:.1f}s"
    assert list(matrix.index) == list(index)
    assert matrix.loc["Gene0", "p0"] == 1
    assert matrix.loc["Gene1", "p0"] == -1
    assert matrix.loc["Gene5", "p5"] == 1  # receptor of p4 is Gene5, but in column p4
    assert matrix.loc["Gene5", "p4"] == -1
    assert int((matrix != 0).to_numpy().sum()) == 2 * N_INTERACTIONS
