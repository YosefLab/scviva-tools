# Developer API

Internal base classes and mixins used to build new scviva models.

## Base classes

```{eval-rst}
.. currentmodule:: scviva.model.base

.. autosummary::
   :toctree: reference/
   :nosignatures:

   SpatialBaseModel
   SpatialNeighborhoodMixin
   SpatialDeconvolutionMixin
   SpatialPredictiveMixin
```

## Data fields

```{eval-rst}
.. currentmodule:: scviva.data

.. autosummary::
   :toctree: reference/
   :nosignatures:

   SpatialCoordsField
   NeighborhoodGraphField
```

## Modules (neural networks)

```{eval-rst}
.. currentmodule:: scviva.module

.. autosummary::
   :toctree: reference/
   :nosignatures:

   nicheVAE
   JVAE
```

## Utilities

```{eval-rst}
.. currentmodule:: scviva.utils

.. autosummary::
   :toctree: reference/
   :nosignatures:

   get_spatial_coords
   validate_spatial_coords
   resolve_device
   stats_dtype
```
