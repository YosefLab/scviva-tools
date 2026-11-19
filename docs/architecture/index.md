# Architecture

`scviva-tools` is built in layers: scvi-tools provides the model foundation, `SpatialBaseModel`
adds the spatial registry and plotting together with the spatial mixins, the models inherit from it,
and downstream tools operate on trained models.

```{image} scviva-tools-poster-diagram.png
:alt: scviva-tools layered architecture overview
:width: 100%
```
