from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS

from scviva.external.amici._constants import AMICI_REGISTRY_KEYS


class AMICIModule(nn.Module):
    """Compact AMICI neighbor-attention module for Phase 1.

    The module keeps the upstream AMICI modeling idea: predict each cell's
    expression from its cell-type mean plus an attention-weighted residual from
    spatial neighbors.
    """

    def __init__(
        self,
        n_genes: int,
        n_labels: int,
        empirical_ct_means: torch.Tensor,
        n_label_embed: int = 16,
        n_nn_embed: int = 32,
        n_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_labels = n_labels
        self.n_label_embed = n_label_embed
        self.n_nn_embed = n_nn_embed
        self.n_hidden = n_hidden

        self.register_buffer("ct_profiles", empirical_ct_means.float())

        self.label_embed = nn.Embedding(n_labels, n_label_embed)
        self.nn_embed = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_nn_embed),
            nn.ReLU(),
        )
        self.query_embed = nn.Linear(n_label_embed, n_nn_embed)
        self.key_embed = nn.Linear(n_nn_embed, n_nn_embed)
        self.value_embed = nn.Linear(n_nn_embed, n_nn_embed)
        self.distance_coef = nn.Linear(n_nn_embed + n_label_embed, 1)
        self.linear_head = nn.Linear(n_nn_embed, n_genes)

    def inference(self, labels: torch.Tensor, nn_x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Embed target labels and neighbor expression."""
        labels = labels.long().view(-1)
        label_embed = self.label_embed(labels)
        nn_embed = self.nn_embed(nn_x.float())
        return {"label_embed": label_embed, "nn_embed": nn_embed}

    def generative(
        self,
        labels: torch.Tensor,
        label_embed: torch.Tensor,
        nn_embed: torch.Tensor,
        nn_dist: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate expression predictions from attention over neighbors."""
        labels = labels.long().view(-1)
        query = self.query_embed(label_embed).unsqueeze(1)
        keys = self.key_embed(nn_embed)
        values = self.value_embed(nn_embed)

        scores = (query * keys).sum(dim=-1) / math.sqrt(self.n_nn_embed)
        repeated_label = label_embed.unsqueeze(1).expand(-1, nn_embed.shape[1], -1)
        distance_coef = F.softplus(
            self.distance_coef(torch.cat([nn_embed, repeated_label], dim=-1))
        ).squeeze(-1)
        scores = scores - distance_coef * nn_dist.float()

        attention_patterns = F.softmax(scores, dim=-1)
        context = torch.sum(attention_patterns.unsqueeze(-1) * values, dim=1)
        residual = self.linear_head(context)
        prediction = self.ct_profiles[labels] + residual
        return {
            "attention_patterns": attention_patterns,
            "nn_embed": nn_embed,
            "residual": residual,
            "prediction": prediction,
        }

    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        compute_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run AMICI inference/generative paths and optionally compute loss."""
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        nn_x = tensors[AMICI_REGISTRY_KEYS.NN_X_KEY]
        nn_dist = tensors[AMICI_REGISTRY_KEYS.NN_DIST_KEY]
        inference_outputs = self.inference(labels, nn_x)
        generative_outputs = self.generative(
            labels,
            inference_outputs["label_embed"],
            inference_outputs["nn_embed"],
            nn_dist,
        )
        if compute_loss:
            true_x = tensors[REGISTRY_KEYS.X_KEY].float()
            generative_outputs["loss"] = F.mse_loss(
                generative_outputs["prediction"],
                true_x,
                reduction="mean",
            )
        return generative_outputs
