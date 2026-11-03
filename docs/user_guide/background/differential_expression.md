# Differential Expression

## Overview

scviva-tools inherits scvi-tools' Bayes-factor differential expression framework and extends
it with **niche-aware DE** (scVIVA) and **niche abundance DE** (ResolVI).

## Standard DE (vanilla / change mode)

All models expose `differential_expression()`. Two modes are supported:

- **vanilla**: computes log-fold-change between groups using posterior samples.
- **change**: computes the posterior probability that the log-fold-change exceeds a threshold
  $\delta$ (default 0.25).

```python
de_df = model.differential_expression(
    adata,
    groupby="cell_type",
    group1=["T cells"],
    group2=["B cells"],
    mode="change",
    delta=0.25,
)
```

## Niche DE (scVIVA)

scVIVA's `differential_niche_expression()` tests for gene expression differences that are
explained by the **cellular microenvironment** (niche composition) rather than intrinsic
cell-type identity. It uses a Gaussian process classifier to attribute expression variance to
niche context.

## Niche abundance DE (ResolVI)

ResolVI's `differential_niche_abundance()` tests for differences in the **composition of the
spatial neighbourhood** between conditions, enabling discovery of altered cell-type co-localisation
patterns across disease states.

## References

- Boyeau et al. (2019) _Deep generative models for detecting differential expression_. bioRxiv.
