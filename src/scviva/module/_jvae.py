"""Joint VAE (JVAE) module for GIMVI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS

if TYPE_CHECKING:
    import numpy as np
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, MultiDecoder, MultiEncoder
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.nn import ModuleList

torch.backends.cudnn.benchmark = True


class JVAE(BaseModuleClass):
    """Joint variational auto-encoder for imputing missing genes in spatial data.

    Implementation of gimVI :cite:p:`Lopez19`.

    Parameters
    ----------
    dim_input_list
        List of number of input genes for each dataset.
    total_genes
        Total number of different genes.
    indices_mappings
        List of mappings from model inputs to model output locations.
    gene_likelihoods
        List of distributions: 'zinb', 'nb', or 'poisson'.
    model_library_bools
        Whether to model library size with a latent variable per dataset.
    library_log_means
        List of 1 x n_batch arrays of log library size means.
    library_log_vars
        List of 1 x n_batch arrays of log library size variances.
    n_latent
        Dimension of latent space.
    n_layers_encoder_individual
        Number of individual encoder layers.
    n_layers_encoder_shared
        Number of shared encoder layers.
    dim_hidden_encoder
        Hidden layer dimension for encoder.
    n_layers_decoder_individual
        Number of individual decoder layers.
    n_layers_decoder_shared
        Number of shared decoder layers.
    dim_hidden_decoder_individual
        Hidden layer dimension for individual decoder.
    dim_hidden_decoder_shared
        Hidden layer dimension for shared decoder.
    dropout_rate_encoder
        Dropout rate for encoder.
    dropout_rate_decoder
        Dropout rate for decoder.
    n_batch
        Total number of batches.
    n_labels
        Total number of labels.
    dispersion
        Dispersion parameterization: 'gene', 'gene-batch', 'gene-label', or 'gene-cell'.
    log_variational
        Log(data+1) prior to encoding for numerical stability.
    """

    def __init__(
        self,
        dim_input_list: list[int],
        total_genes: int,
        indices_mappings: list[np.ndarray | slice],
        gene_likelihoods: list[str],
        model_library_bools: list[bool],
        library_log_means: list[np.ndarray | None],
        library_log_vars: list[np.ndarray | None],
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 1,
        dim_hidden_encoder: int = 64,
        n_layers_decoder_individual: int = 0,
        n_layers_decoder_shared: int = 0,
        dim_hidden_decoder_individual: int = 64,
        dim_hidden_decoder_shared: int = 64,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float = 0.2,
        n_batch: int = 0,
        n_labels: int = 0,
        dispersion: str = "gene-batch",
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.indices_mappings = indices_mappings
        self.gene_likelihoods = gene_likelihoods
        self.model_library_bools = model_library_bools
        for mode in range(len(dim_input_list)):
            if self.model_library_bools[mode]:
                self.register_buffer(
                    f"library_log_means_{mode}",
                    torch.from_numpy(library_log_means[mode]).float(),
                )
                self.register_buffer(
                    f"library_log_vars_{mode}",
                    torch.from_numpy(library_log_vars[mode]).float(),
                )

        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.dispersion = dispersion
        self.log_variational = log_variational

        self.z_encoder = MultiEncoder(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
            return_dist=True,
        )

        self.l_encoders = ModuleList(
            [
                Encoder(
                    self.n_input_list[i],
                    1,
                    n_layers=1,
                    dropout_rate=dropout_rate_encoder,
                    return_dist=True,
                )
                if self.model_library_bools[i]
                else None
                for i in range(len(self.n_input_list))
            ]
        )

        self.decoder = MultiDecoder(
            self.n_latent,
            self.total_genes,
            n_hidden_conditioned=dim_hidden_decoder_individual,
            n_hidden_shared=dim_hidden_decoder_shared,
            n_layers_conditioned=n_layers_decoder_individual,
            n_layers_shared=n_layers_decoder_shared,
            n_cat_list=[self.n_batch],
            dropout_rate=dropout_rate_decoder,
        )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_labels))

    def sample_from_posterior_z(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            Tensor of counts with shape ``(n_cells, n_genes_mode)``, where ``n_genes_mode`` is
            the number of genes of dataset ``mode``.
        mode
            Index of the dataset (encoder head) ``x`` comes from. May be ``None`` only when the
            model has a single dataset, in which case ``0`` is used.
        deterministic
            If ``True``, return the posterior mean instead of a sample.

        Returns
        -------
        Tensor of latent values with shape ``(n_cells, n_latent)``.
        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode when having multiple datasets")
        outputs = self.inference(x, mode)
        qz_m = outputs["qz"].loc
        z = outputs["z"]
        if deterministic:
            z = qz_m
        return z

    def sample_from_posterior_l(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            Tensor of counts with shape ``(n_cells, n_genes_mode)``, where ``n_genes_mode`` is
            the number of genes of dataset ``mode``.
        mode
            Index of the dataset (encoder head) ``x`` comes from.
        deterministic
            If ``True`` and the library size is modeled for this dataset, return the posterior
            mean of the log library size instead of a sample. If the library size is not
            modeled, the observed log library size is returned regardless.

        Returns
        -------
        Tensor of log library sizes with shape ``(n_cells, 1)``.
        """
        inference_out = self.inference(x, mode)
        return (
            inference_out["ql"].loc
            if (deterministic and inference_out["ql"] is not None)
            else inference_out["library"]
        )

    def sample_scale(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: torch.Tensor | None = None,
        deterministic: bool = False,
        decode_mode: int | None = None,
    ) -> torch.Tensor:
        """Return the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            Tensor of counts with shape ``(n_cells, n_genes_mode)``, where ``n_genes_mode`` is
            the number of genes of dataset ``mode``.
        mode
            Index of the dataset (encoder head) ``x`` comes from.
        batch_index
            Batch index of each cell, with shape ``(n_cells, 1)``.
        y
            Label of each cell, with shape ``(n_cells, 1)``. Required when
            ``dispersion="gene-label"``.
        deterministic
            If ``True``, use the posterior means of ``z`` and of the library size (when the
            library is modeled) instead of samples.
        decode_mode
            Index of the dataset whose decoder head is used. If ``None``, defaults to ``mode``.

        Returns
        -------
        Tensor of normalized expression frequencies with shape ``(n_cells, total_genes)``.
        """
        gen_out = self._run_forward(
            x, mode, batch_index, y=y, deterministic=deterministic, decode_mode=decode_mode
        )
        return gen_out["px_scale"]

    def get_sample_rate(self, x, batch_index, *_, **__):
        """Get the sample rate for the model.

        Encodes and decodes ``x`` with dataset ``0`` (see :meth:`sample_rate`).

        Parameters
        ----------
        x
            Tensor of counts for dataset ``0`` with shape ``(n_cells, n_genes_0)``.
        batch_index
            Batch index of each cell, with shape ``(n_cells, 1)``.
        *_
            Ignored; accepted for interface compatibility.
        **__
            Ignored; accepted for interface compatibility.
        """
        return self.sample_rate(x, 0, batch_index)

    def _run_forward(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: torch.Tensor | None = None,
        deterministic: bool = False,
        decode_mode: int = None,
    ) -> dict:
        """Run the forward pass of the model."""
        if decode_mode is None:
            decode_mode = mode
        inference_out = self.inference(x, mode)
        if deterministic:
            z = inference_out["qz"].loc
            if inference_out["ql"] is not None:
                library = inference_out["ql"].loc
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            library = inference_out["library"]
        gen_out = self.generative(z, library, batch_index, y, decode_mode)
        return gen_out

    def sample_rate(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: torch.Tensor | None = None,
        deterministic: bool = False,
        decode_mode: int = None,
    ) -> torch.Tensor:
        """Return the tensor of scaled frequencies of expression.

        Parameters
        ----------
        x
            Tensor of counts with shape ``(n_cells, n_genes_mode)``, where ``n_genes_mode`` is
            the number of genes of dataset ``mode``.
        mode
            Index of the dataset (encoder head) ``x`` comes from.
        batch_index
            Batch index of each cell, with shape ``(n_cells, 1)``.
        y
            Label of each cell, with shape ``(n_cells, 1)``. Required when
            ``dispersion="gene-label"``.
        deterministic
            If ``True``, use the posterior means of ``z`` and of the library size (when the
            library is modeled) instead of samples.
        decode_mode
            Index of the dataset whose decoder head is used. If ``None``, defaults to ``mode``.

        Returns
        -------
        Tensor of expected counts (``px_scale`` times library size) with shape
        ``(n_cells, total_genes)``.
        """
        gen_out = self._run_forward(
            x, mode, batch_index, y=y, deterministic=deterministic, decode_mode=decode_mode
        )
        return gen_out["px_rate"]

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        """Compute the reconstruction loss.

        Parameters
        ----------
        x
            Tensor of counts with shape ``(n_cells, n_genes_mode)``, where ``n_genes_mode`` is
            the number of genes of dataset ``mode``.
        px_rate
            Mean of the count distribution, restricted to the genes of dataset ``mode``.
        px_r
            Inverse dispersion of the count distribution, restricted to the genes of dataset
            ``mode``.
        px_dropout
            Logits of the zero-inflation probability, restricted to the genes of dataset
            ``mode``. Only used when the gene likelihood is ``"zinb"``.
        mode
            Index of the dataset; selects the gene likelihood from ``gene_likelihoods``.

        Returns
        -------
        Negative log-likelihood per cell with shape ``(n_cells,)``.
        """
        reconstruction_loss = None
        if self.gene_likelihoods[mode] == "zinb":
            reconstruction_loss = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "nb":
            reconstruction_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
        elif self.gene_likelihoods[mode] == "poisson":
            reconstruction_loss = -Poisson(px_rate).log_prob(x).sum(dim=1)
        return reconstruction_loss

    def _get_inference_input(self, tensors) -> dict[str, torch.Tensor | None]:
        """Get the input for the inference model."""
        return {
            "x": tensors[REGISTRY_KEYS.X_KEY],
            "batch_index": tensors.get(REGISTRY_KEYS.BATCH_KEY, None),
        }

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        """Get the input for the generative model."""
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        return {"z": z, "library": library, "batch_index": batch_index, "y": y}

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        mode: int | None = 0,
        n_samples: int | None = 1,
        batch_index: torch.Tensor | None = None,
    ) -> dict:
        """Run the inference model.

        Parameters
        ----------
        x
            Tensor of counts with shape ``(n_cells, n_genes_mode)``, where ``n_genes_mode`` is
            the number of genes of dataset ``mode``.
        mode
            Index of the dataset (encoder head) ``x`` comes from.
        n_samples
            Number of samples of ``z`` to draw from the posterior. If greater than ``1``,
            ``z`` gains a leading sample dimension.
        batch_index
            Batch index of each cell. Unused; accepted for interface compatibility.

        Returns
        -------
        Dictionary with keys ``"qz"`` (posterior over ``z``), ``"z"``, ``"ql"`` (posterior
        over the log library size, or ``None`` if the library is not modeled) and
        ``"library"`` (sampled or observed log library size).
        """
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz, z = self.z_encoder(x_, mode)
        ql, library = None, None
        if self.model_library_bools[mode]:
            ql, library = self.l_encoders[mode](x_)
        else:
            library = torch.log(torch.sum(x, dim=1)).view(-1, 1)

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)

        return {"qz": qz, "z": z, "ql": ql, "library": library}

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        mode: int | None = 0,
        transform_batch: torch.Tensor | None = None,
    ) -> dict:
        """Run the generative model.

        Parameters
        ----------
        z
            Latent values with shape ``(n_cells, n_latent)``.
        library
            Log library size with shape ``(n_cells, 1)``.
        batch_index
            Batch index of each cell, with shape ``(n_cells, 1)``. Used by the decoder and,
            when ``dispersion="gene-batch"``, to select the dispersion.
        y
            Label of each cell, with shape ``(n_cells, 1)``. Used when
            ``dispersion="gene-label"`` to select the dispersion.
        mode
            Index of the dataset whose decoder head is used. ``px_scale`` is normalized over
            the genes of this dataset.
        transform_batch
            If not ``None``, the ``"batch_index"`` entry of the returned dictionary is replaced
            by this value. It does not affect the decoded quantities.

        Returns
        -------
        Dictionary with keys ``"px_scale"``, ``"px"``, ``"px_r"``, ``"px_rate"``,
        ``"px_dropout"`` and ``"batch_index"``.
        """
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            z, mode, library, self.dispersion, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(F.one_hot(y.squeeze(-1).long(), self.n_labels).float(), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r.view(1, self.px_r.size(0))
        px_r = torch.exp(px_r)

        px_scale = px_scale / torch.sum(px_scale[:, self.indices_mappings[mode]], dim=1).view(
            -1, 1
        )
        px_rate = px_scale * torch.exp(library)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)

        return {
            "px_scale": px_scale,
            "px": px,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            "batch_index": batch_index,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        mode: int | None = None,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        tensors
            Dictionary of minibatch tensors; uses the registered counts and batch index.
        inference_outputs
            Output of :meth:`inference`.
        generative_outputs
            Output of :meth:`generative`.
        mode
            Index of the dataset the minibatch comes from. May be ``None`` only when the
            model has a single dataset, in which case ``0`` is used.
        kl_weight
            Weight of the KL divergence terms (``z`` and, if modeled, library size).

        Returns
        -------
        :class:`~scvi.module.base.LossOutput` with the total loss, per-cell reconstruction
        loss and per-cell local KL divergence.
        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode")
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz = inference_outputs["qz"]
        ql = inference_outputs["ql"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )

        mean = torch.zeros_like(qz.loc)
        scale = torch.ones_like(qz.scale)
        kl_divergence_z = kl(qz, Normal(mean, scale)).sum(dim=1)

        if self.model_library_bools[mode]:
            library_log_means = getattr(self, f"library_log_means_{mode}")
            library_log_vars = getattr(self, f"library_log_vars_{mode}")

            local_library_log_means = F.linear(
                F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), library_log_means
            )
            local_library_log_vars = F.linear(
                F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), library_log_vars
            )
            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local = kl_divergence_l + kl_divergence_z
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossOutput(loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_local)
