from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from scviva.data._fields import SpatialCoordsField
from scviva.external.starfysh._module import StarfyshModule
from scviva.model.base._deconvolution_mixin import SpatialDeconvolutionMixin
from scviva.model.base._spatial_base import SpatialBaseModel

if TYPE_CHECKING:
    from anndata import AnnData


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


class Starfysh(SpatialDeconvolutionMixin, SpatialBaseModel):
    """Phase 1 expression-only Starfysh wrapper."""

    _module_cls = StarfyshModule

    def __init__(
        self,
        adata: AnnData,
        signature_scores: pd.DataFrame | np.ndarray,
        cell_type_names: list[str] | None = None,
        **model_kwargs,
    ) -> None:
        super().__init__(adata)
        signatures = self._format_signature_scores(signature_scores, adata.n_obs, cell_type_names)
        self.signature_scores = signatures.to_numpy(dtype=np.float32)
        self.cell_type_mapping = np.asarray(signatures.columns)

        self.module = self._module_cls(
            n_genes=self.summary_stats.n_vars,
            n_cell_types=self.signature_scores.shape[1],
            **model_kwargs,
        )
        self._device = torch.device("cpu")
        self._model_summary_string = (
            f"Starfysh Model with params: n_genes={self.summary_stats.n_vars}, "
            f"n_cell_types={self.signature_scores.shape[1]}"
        )
        self.init_params_ = self._get_init_params(locals())

    @staticmethod
    def _format_signature_scores(
        signature_scores: pd.DataFrame | np.ndarray,
        n_obs: int,
        cell_type_names: list[str] | None,
    ) -> pd.DataFrame:
        if isinstance(signature_scores, pd.DataFrame):
            scores = signature_scores.copy()
        else:
            values = np.asarray(signature_scores, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError("signature_scores must be a 2D array or DataFrame.")
            columns = cell_type_names or [f"cell_type_{i}" for i in range(values.shape[1])]
            scores = pd.DataFrame(values, columns=columns)
        if scores.shape[0] != n_obs:
            raise ValueError(
                "signature_scores must have one row per observation in the registered AnnData."
            )
        if np.any(scores.to_numpy() < 0):
            raise ValueError("signature_scores must be non-negative.")
        row_sums = scores.sum(axis=1).replace(0, np.nan)
        scores = scores.div(row_sums, axis=0).fillna(1.0 / scores.shape[1])
        return scores

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Register expression and spatial fields for Starfysh."""
        setup_method_args = cls._get_setup_method_args(**locals())
        fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            SpatialCoordsField(obsm_key=spatial_key),
        ]
        adata_manager = AnnDataManager(fields=fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def _tensor_dataset(self) -> TensorDataset:
        x = _as_numpy(self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)).astype(np.float32)
        library = np.log1p(x.sum(axis=1, keepdims=True)).astype(np.float32)
        return TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(self.signature_scores, dtype=torch.float32),
            torch.as_tensor(library, dtype=torch.float32),
        )

    def _batch_to_tensors(self, batch) -> dict[str, torch.Tensor]:
        x, signature_scores, library = (item.to(self._device) for item in batch)
        return {
            REGISTRY_KEYS.X_KEY: x,
            "signature_scores": signature_scores,
            "library": library,
        }

    def _collect_outputs(
        self,
        keys: tuple[str, ...],
        batch_size: int,
        prog_bar: bool = False,
    ) -> dict[str, np.ndarray]:
        self._check_if_trained(warn=True)
        self.module.eval()
        loader = DataLoader(self._tensor_dataset(), batch_size=batch_size, shuffle=False)
        outputs_by_key: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        with torch.no_grad():
            for batch in tqdm(loader, disable=not prog_bar):
                tensors = self._batch_to_tensors(batch)
                outputs = self.module(tensors, compute_loss=False)
                for key in keys:
                    outputs_by_key[key].append(outputs[key].detach().cpu().numpy())
        return {key: np.concatenate(values, axis=0) for key, values in outputs_by_key.items()}

    def train(
        self,
        max_epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        prog_bar: bool = False,
    ) -> None:
        """Train the Phase 1 expression-only Starfysh module."""
        self._device = torch.device(device)
        self.module.to(self._device)
        self.module.train()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)
        dataset = self._tensor_dataset()
        if len(dataset) < 2:
            raise ValueError("Starfysh training requires at least two observations.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        train_batch_size = min(max(batch_size, 2), len(dataset))
        if len(dataset) % train_batch_size == 1:
            train_batch_size = min(train_batch_size + 1, len(dataset))
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        history = []
        for _ in tqdm(range(max_epochs), disable=not prog_bar):
            losses = []
            for batch in loader:
                tensors = self._batch_to_tensors(batch)
                outputs = self.module(tensors, compute_loss=True)
                loss = outputs["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(np.mean(losses))

        self.history_ = {"loss": history}
        self.is_trained_ = True
        self.module.eval()
        return None

    def get_proportions(
        self,
        adata: AnnData | None = None,
        batch_size: int = 128,
        store_key: str | None = None,
        prog_bar: bool = False,
    ) -> pd.DataFrame:
        """Return Starfysh cell-type proportions for the registered AnnData."""
        if adata is not None and adata is not self.adata:
            raise ValueError("Phase 1 Starfysh only supports the registered AnnData.")

        proportions = self._collect_outputs(
            ("qc_m",),
            batch_size=batch_size,
            prog_bar=prog_bar,
        )["qc_m"]
        if store_key is not None:
            self.adata.obsm[store_key] = proportions
        return pd.DataFrame(
            proportions,
            index=self.adata.obs_names,
            columns=self.cell_type_mapping,
        )

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: np.ndarray | list[int] | None = None,
        give_mean: bool = True,
        batch_size: int = 128,
        store_key: str | None = None,
        prog_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Return Starfysh latent expression representation for the registered AnnData."""
        if kwargs:
            raise TypeError(f"Unsupported keyword arguments: {sorted(kwargs)}")
        if adata is not None and adata is not self.adata:
            raise ValueError("Phase 2A Starfysh only supports the registered AnnData.")

        key = "qz_m" if give_mean else "qz"
        latent = self._collect_outputs((key,), batch_size=batch_size, prog_bar=prog_bar)[key]
        if store_key is not None:
            self.adata.obsm[store_key] = latent
        if indices is not None:
            latent = latent[np.asarray(indices)]
        return latent

    def get_model_outputs(
        self,
        batch_size: int = 128,
        store: bool = False,
        prog_bar: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return Starfysh inference and generative outputs for the registered AnnData."""
        outputs = self._collect_outputs(
            ("qc_m", "qz_m", "qz_logv", "px_rate", "px_scale"),
            batch_size=batch_size,
            prog_bar=prog_bar,
        )
        if store:
            self.adata.obsm["starfysh_proportions"] = outputs["qc_m"]
            self.adata.obsm["X_starfysh"] = outputs["qz_m"]
            self.adata.layers["starfysh_px_rate"] = outputs["px_rate"]
        return outputs
