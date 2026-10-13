# User API

Import scVIVA-Tools as:

```python
import scviva
```

## Models

```{eval-rst}
.. currentmodule:: scviva

.. autosummary::
   :toctree: reference/
   :nosignatures:

   model.SCVIVA
   model.DestVI
   model.ResolVI
   model.GIMVI
```

## External models

```{eval-rst}
.. currentmodule:: scviva

.. autosummary::
   :toctree: reference/
   :nosignatures:

   external.DIAGVI
   external.RNAStereoscope
   external.SpatialStereoscope
   external.Tangram
```

## Tools

```{eval-rst}
.. currentmodule:: scviva

.. autosummary::
   :toctree: reference/
   :nosignatures:

   tl.HarremanAnalysis
   tl.VisionAnalysis
```

### Analysis accessors

`HarremanAnalysis` and `VisionAnalysis` group lower-level functions under accessors
(`ha.hs`, `ha.vs`, `ha.tl`, `ha.pl` and `va.tl`). Each accessor method takes the same
arguments as the function it forwards to, with `adata` defaulting to the analysis
object's AnnData.

```{eval-rst}
.. currentmodule:: scviva.tools

.. autosummary::
   :toctree: reference/
   :nosignatures:

   harreman.HarremanHotspotAccessor
   harreman.HarremanVisionAccessor
   harreman.HarremanToolsAccessor
   harreman.HarremanPlottingAccessor
   vision.VisionToolsAccessor
```

### Harreman functions

```{eval-rst}
.. currentmodule:: scviva.tools.harreman

.. rubric:: Hotspot (``hs``)

.. autosummary::
   :toctree: reference/
   :nosignatures:

   hotspot.compute_local_autocorrelation
   hotspot.compute_local_correlation
   hotspot.create_modules
   hotspot.calculate_module_scores
   hotspot.calculate_super_module_scores
   hotspot.compute_top_scoring_modules
   hotspot.load_metabolic_genes
   vision.integrate_vision_hotspot_results

.. rubric:: Cell-cell communication tools (``tl``)

.. autosummary::
   :toctree: reference/
   :nosignatures:

   tools.compute_knn_graph
   tools.apply_gene_filtering
   tools.compute_gene_pairs
   tools.compute_cell_communication
   tools.compute_ct_cell_communication
   tools.compute_interacting_cell_scores
   tools.compute_ct_interacting_cell_scores
   tools.select_significant_interactions
   tools.compute_interaction_module_correlation

.. rubric:: Preprocessing (``pp``)

.. autosummary::
   :toctree: reference/
   :nosignatures:

   preprocessing.extract_interaction_db
   preprocessing.setup_deconv_adata
   preprocessing.read_h5ad
   preprocessing.write_h5ad

.. rubric:: Datasets (``ds``)

.. autosummary::
   :toctree: reference/
   :nosignatures:

   datasets.load_visium_mouse_colon_dataset
   datasets.load_slide_seq_human_lung_dataset
```

### VISION functions

```{eval-rst}
.. currentmodule:: scviva.tools.vision

.. rubric:: Results and data containers

.. autosummary::
   :toctree: reference/
   :nosignatures:

   VisionResults
   NormData

.. rubric:: Preprocessing

.. autosummary::
   :toctree: reference/
   :nosignatures:

   apply_filters
   filter_genes_fano
   filter_genes_novar
   filter_genes_threshold
   get_normalized_copy
   get_normalized_copy_sparse
   log2p1

.. rubric:: Latent space and projections

.. autosummary::
   :toctree: reference/
   :nosignatures:

   compute_latent_space
   apply_pca
   apply_permutation_wpca
   apply_ica
   apply_isomap
   apply_rbfpca
   apply_tsne
   apply_umap
   generate_projections

.. rubric:: Neighbor graphs

.. autosummary::
   :toctree: reference/
   :nosignatures:

   find_knn
   compute_knn_weights
   compute_knn_weights_anndata
   compute_knn_weights_from_tree
   compute_knn_weights_from_tree_anndata
   compute_knn_weights_from_tree_lca
   compute_knn_weights_from_tree_lca_anndata

.. rubric:: Micro-clustering

.. autosummary::
   :toctree: reference/
   :nosignatures:

   apply_micro_clustering
   pool_matrix
   pool_matrix_anndata
   pool_metadata
   pool_metadata_anndata

.. rubric:: Signatures

.. autosummary::
   :toctree: reference/
   :nosignatures:

   load_signatures
   split_signed_signatures
   compute_signature_scores
   compute_signatures_anndata
   compute_obs_df_scores

.. rubric:: Phylogeny

.. autosummary::
   :toctree: reference/
   :nosignatures:

   cluster_cells_tree
   compute_plasticity_scores

.. rubric:: Differential expression

.. autosummary::
   :toctree: reference/
   :nosignatures:

   rank_genes_groups
```

## Plotting

```{eval-rst}
.. currentmodule:: scviva

.. autosummary::
   :toctree: reference/
   :nosignatures:

   pl.harreman.local_correlation_plot
   pl.harreman.average_local_correlation_plot
   pl.harreman.module_score_correlation_plot
   pl.harreman.plot_interacting_cell_scores
   pl.harreman.plot_ct_interacting_cell_scores
   pl.harreman.plot_interaction_module_correlation
   pl.vision.plot_vision_autocorrelation
   pl.vision.plot_vision_de_results
   pl.vision.plot_signature_for_selection
   pl.vision.plot_selection_histplot
   pl.plot_cell_type_map
   pl.scviva_de.plot_niche_de_decision_boundary
   pl.scviva_de.plot_connectivity_distribution
```

## Settings

```{eval-rst}
.. currentmodule:: scviva

.. autosummary::
   :toctree: reference/
   :nosignatures:

   settings
```
