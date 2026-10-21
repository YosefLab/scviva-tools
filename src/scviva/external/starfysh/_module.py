from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS


class StarfyshModule(nn.Module):
    """Compact expression-only Starfysh AVAE module for Phase 1."""

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        alpha_mul: float = 50.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.alpha_mul = alpha_mul
        self.eps = eps

        self.cell_encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Linear(n_hidden, n_cell_types),
            nn.Softmax(dim=-1),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )
        self.latent_mean = nn.Linear(n_hidden, n_latent)
        self.latent_logvar = nn.Linear(n_hidden, n_latent)
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes),
            nn.Softmax(dim=-1),
        )
        self.px_r = nn.Parameter(torch.randn(n_genes))

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the diagonal latent posterior."""
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Infer cell-type proportions and latent expression state."""
        x_n = torch.log1p(x.float())
        qc_m = self.cell_encoder(x_n)
        hidden = self.latent_encoder(x_n)
        qz_m = self.latent_mean(hidden)
        qz_logv = self.latent_logvar(hidden)
        qz = self.reparameterize(qz_m, qz_logv)
        return {"qc_m": qc_m, "qz_m": qz_m, "qz_logv": qz_logv, "qz": qz}

    def generative(
        self,
        inference_outputs: dict[str, torch.Tensor],
        library: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate expression rates from latent state and library size."""
        px_scale = self.decoder(inference_outputs["qz"])
        px_rate = torch.exp(library.float()) * px_scale + self.eps
        return {
            "px_rate": px_rate,
            "px_scale": px_scale,
            "px_r": F.softplus(self.px_r) + self.eps,
        }

    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        compute_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the Starfysh expression path and optionally compute loss."""
        x = tensors[REGISTRY_KEYS.X_KEY].float()
        signature_scores = tensors["signature_scores"].float()
        library = tensors["library"].float()

        inference_outputs = self.inference(x)
        generative_outputs = self.generative(inference_outputs, library)
        outputs = {**inference_outputs, **generative_outputs}

        if compute_loss:
            recon_loss = F.mse_loss(torch.log1p(generative_outputs["px_rate"]), torch.log1p(x))
            signature_loss = F.mse_loss(inference_outputs["qc_m"], signature_scores)
            outputs["loss"] = recon_loss + self.alpha_mul * 0.01 * signature_loss
        return outputs
