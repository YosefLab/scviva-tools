# Variational Inference

## The generative model

All scviva-tools models are based on **amortized variational inference** (AVI). The core idea
is to learn a probabilistic generative model $p_\theta(x \mid z)$ of observed gene expression $x$
conditioned on a low-dimensional latent variable $z$, alongside an approximate posterior
$q_\phi(z \mid x)$ parameterized by an encoder neural network.

Training maximizes the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - \mathrm{KL}(q_\phi(z \mid x) \| p(z))$$

## Gene likelihood

Most models support two gene likelihood distributions:

- **Negative Binomial (NB)**: default; models overdispersed count data.
- **Poisson**: simpler; suitable for very low-count spatial data.

## Spatial priors (scVIVA, ResolVI)

Spatial models extend the standard VAE with a **niche-aware prior** that conditions the latent
distribution on the cellular neighbourhood, encoding microenvironment structure directly into the
latent space.

## KL annealing

To stabilize early training, the KL divergence term is annealed from 0 to 1 over the first
`n_epochs_kl_warmup` epochs. This prevents the model from collapsing to the prior before the
encoder has learned a meaningful representation.

## References

- Lopez et al. (2018) _Deep generative modeling for single-cell transcriptomics_. Nature Methods.
- Levy et al. (2025) _scVIVA_. bioRxiv.
