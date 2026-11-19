"""LR interactions are grouped under their CellChat pathway in ``gene_pair_dict``.

``run_cell_communication_analysis`` relabels each ligand-receptor "metabolite" with its
``pathway_name`` from ``uns["LR_database"]``. It used a chained assignment
(``df["metabolite"][mask] = ...``), which is a silent no-op under pandas 3 copy-on-write,
so interactions were never grouped by pathway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from scviva.tools import HarremanAnalysis
from scviva.tools.harreman.tools.cell_communication import (
    compute_gene_pairs,
    run_cell_communication_analysis,
)


def _lr_adata_with_pathways() -> AnnData:
    n_obs = 40
    genes = [f"gene{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(2.0, size=(n_obs, len(genes))).astype(float),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=genes),
        obsm={"spatial": rng.uniform(0, 100, size=(n_obs, 2))},
    )
    pairs = ["pair0", "pair1", "pair2"]
    database = pd.DataFrame(0.0, index=genes, columns=pairs)
    for k, pair in enumerate(pairs):
        database.loc[f"gene{2 * k}", pair] = 1.0  # ligand
        database.loc[f"gene{2 * k + 1}", pair] = -1.0  # receptor
    adata.varm["database"] = database
    adata.uns["database"] = "LR"
    adata.uns["ligand"] = pd.DataFrame({0: ["gene0", "gene2", "gene4"]}, index=pairs)
    adata.uns["receptor"] = pd.DataFrame({0: ["gene1", "gene3", "gene5"]}, index=pairs)
    # pair0 and pair1 belong to the same pathway; pair2 has no pathway annotation.
    adata.uns["LR_database"] = pd.DataFrame(
        {
            "interaction_name": pairs,
            "pathway_name": ["PATHWAY_X", "PATHWAY_X", np.nan],
            "annotation": ["Secreted Signaling"] * 3,
        },
        index=pairs,
    )
    HarremanAnalysis(adata).tl.compute_knn_graph(compute_neighbors_on_key="spatial", n_neighbors=5)
    compute_gene_pairs(adata, cell_type_key=None, ct_specific=False, verbose=False)
    return adata


def test_lr_interactions_are_grouped_by_pathway_in_gene_pair_dict():
    adata = _lr_adata_with_pathways()
    adata.uns["ccc_results"] = {}  # normally initialized by compute_cell_communication

    run_cell_communication_analysis(
        adata,
        layer_key_p_test=None,
        layer_key_np_test=None,
        model="none",
        center_counts_for_np_test=False,
        subset_gene_pairs=None,
        M=5,
        seed=0,
        test="parametric",
        mean="algebraic",
        check_analytic_null=False,
        device="cpu",
        verbose=False,
    )

    gene_pair_dict = adata.uns["gene_pair_dict"]
    assert set(gene_pair_dict) == {"PATHWAY_X", "pair2"}
    assert len(gene_pair_dict["PATHWAY_X"]) == 2
    assert len(gene_pair_dict["pair2"]) == 1
