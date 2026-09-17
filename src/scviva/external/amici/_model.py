from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, ObsmField
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from scviva.external.amici._constants import AMICI_REGISTRY_KEYS
from scviva.external.amici._module import AMICIModule
from scviva.model.base._spatial_base import SpatialBaseModel

if TYPE_CHECKING:
    from anndata import AnnData


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


class AMICI(SpatialBaseModel):
    """Phase 1 AMICI wrapper for neighborhood-aware expression prediction."""

    _module_cls = AMICIModule

    def __init__(self, adata: AnnData, **model_kwargs) -> None:
        super().__init__(adata)
        x = _as_numpy(self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)).astype(np.float32)
        labels = (
            np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY))
            .reshape(-1)
            .astype(int)
        )
        n_labels = int(labels.max()) + 1

        empirical_ct_means = []
        for label_idx in range(n_labels):
            label_mask = labels == label_idx
            if not np.any(label_mask):
                empirical_ct_means.append(np.zeros(x.shape[1], dtype=np.float32))
            else:
                empirical_ct_means.append(x[label_mask].mean(axis=0))
        empirical_ct_means_t = torch.as_tensor(np.vstack(empirical_ct_means), dtype=torch.float32)

        self.module = self._module_cls(
            n_genes=x.shape[1],
            n_labels=n_labels,
            empirical_ct_means=empirical_ct_means_t,
            **model_kwargs,
        )
        self._device = torch.device("cpu")
        self._model_summary_string = (
            f"AMICI Model with params: n_genes={x.shape[1]}, n_labels={n_labels}"
        )
        self.init_params_ = self._get_init_params(locals())

    @property
    def device(self) -> torch.device:
        """Torch device for the wrapped module."""
        return self._device

    @staticmethod
    def _compute_nn(
        adata: AnnData,
        coord_obsm_key: str,
        index_key: str,
        dist_key: str,
        n_neighbors: int,
        labels_obs_key: str,
        exclude_self_labels: bool = True,
    ) -> None:
        coords = np.asarray(adata.obsm[coord_obsm_key], dtype=np.float32)
        labels = adata.obs[labels_obs_key].to_numpy()

        nn_idx = np.zeros((adata.n_obs, n_neighbors), dtype=np.int64)
        nn_dist = np.zeros((adata.n_obs, n_neighbors), dtype=np.float32)

        if not exclude_self_labels:
            n_fit_neighbors = min(n_neighbors + 1, adata.n_obs)
            nn = NearestNeighbors(n_neighbors=n_fit_neighbors, metric="euclidean").fit(coords)
            dist, idx = nn.kneighbors(coords, return_distance=True)
            dist, idx = dist[:, 1:], idx[:, 1:]
            if idx.shape[1] < n_neighbors:
                raise ValueError("n_neighbors is too large for this AnnData.")
            adata.obsm[index_key] = idx[:, :n_neighbors].astype(np.int64)
            adata.obsm[dist_key] = dist[:, :n_neighbors].astype(np.float32)
            return

        for label in np.unique(labels):
            query_idx = np.where(labels == label)[0]
            candidate_idx = np.where(labels != label)[0]
            if len(candidate_idx) < n_neighbors:
                raise ValueError(
                    "n_neighbors is too large after excluding same-label neighbors. "
                    "Use fewer neighbors or set exclude_self_labels=False."
                )
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(
                coords[candidate_idx]
            )
            dist, idx = nn.kneighbors(coords[query_idx], return_distance=True)
            nn_idx[query_idx] = candidate_idx[idx]
            nn_dist[query_idx] = dist

        adata.obsm[index_key] = nn_idx
        adata.obsm[dist_key] = nn_dist

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        labels_key: str | None = None,
        spatial_key: str = "spatial",
        nn_idx_key: str = AMICI_REGISTRY_KEYS.NN_IDX_KEY,
        nn_dist_key: str = AMICI_REGISTRY_KEYS.NN_DIST_KEY,
        n_neighbors: int = 30,
        exclude_self_labels: bool = True,
        **kwargs,
    ) -> None:
        """Register AnnData fields and compute AMICI neighbor arrays."""
        if labels_key is None:
            raise ValueError("labels_key is required for AMICI setup_anndata.")
        if spatial_key not in adata.obsm:
            raise KeyError(f"'{spatial_key}' not found in adata.obsm.")

        cls._compute_nn(
            adata,
            coord_obsm_key=spatial_key,
            index_key=nn_idx_key,
            dist_key=nn_dist_key,
            n_neighbors=n_neighbors,
            labels_obs_key=labels_key,
            exclude_self_labels=exclude_self_labels,
        )

        setup_method_args = cls._get_setup_method_args(**locals())
        fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            ObsmField(AMICI_REGISTRY_KEYS.COORD_KEY, spatial_key),
            ObsmField(AMICI_REGISTRY_KEYS.NN_IDX_KEY, nn_idx_key),
            ObsmField(AMICI_REGISTRY_KEYS.NN_DIST_KEY, nn_dist_key),
        ]
        adata_manager = AnnDataManager(fields=fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def _tensor_dataset(self) -> TensorDataset:
        x = _as_numpy(self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)).astype(np.float32)
        labels = (
            np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY))
            .reshape(-1)
            .astype(np.int64)
        )
        nn_idx = np.asarray(
            self.adata_manager.get_from_registry(AMICI_REGISTRY_KEYS.NN_IDX_KEY)
        ).astype(np.int64)
        nn_dist = np.asarray(
            self.adata_manager.get_from_registry(AMICI_REGISTRY_KEYS.NN_DIST_KEY)
        ).astype(np.float32)
        nn_x = x[nn_idx]
        return TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
            torch.as_tensor(nn_x, dtype=torch.float32),
            torch.as_tensor(nn_dist, dtype=torch.float32),
        )

    def _batch_to_tensors(self, batch) -> dict[str, torch.Tensor]:
        x, labels, nn_x, nn_dist = (item.to(self._device) for item in batch)
        return {
            REGISTRY_KEYS.X_KEY: x,
            REGISTRY_KEYS.LABELS_KEY: labels,
            AMICI_REGISTRY_KEYS.NN_X_KEY: nn_x,
            AMICI_REGISTRY_KEYS.NN_DIST_KEY: nn_dist,
        }

    def _collect_outputs(
        self,
        keys: tuple[str, ...],
        batch_size: int,
        prog_bar: bool,
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
        max_epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        prog_bar: bool = False,
    ):
        """Train AMICI with a small torch loop for the Phase 1 surface."""
        self._device = torch.device(device)
        self.module.to(self._device)
        self.module.train()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)
        loader = DataLoader(self._tensor_dataset(), batch_size=batch_size, shuffle=True)

        history = []
        epoch_iter = tqdm(range(max_epochs), disable=not prog_bar)
        for _ in epoch_iter:
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

    def get_predictions(
        self,
        batch_size: int = 128,
        get_residuals: bool = False,
        store_key: str | None = None,
        prog_bar: bool = False,
    ) -> np.ndarray:
        """Return AMICI predictions or residuals for the registered AnnData."""
        key = "residual" if get_residuals else "prediction"
        output = self._collect_outputs((key,), batch_size=batch_size, prog_bar=prog_bar)[key]
        if store_key is not None:
            self.adata.obsm[store_key] = output
        return output

    def get_attention_patterns(
        self,
        batch_size: int = 128,
        store_key: str | None = None,
        prog_bar: bool = False,
    ) -> np.ndarray:
        """Return AMICI neighbor attention patterns for the registered AnnData."""
        output = self._collect_outputs(
            ("attention_patterns",),
            batch_size=batch_size,
            prog_bar=prog_bar,
        )["attention_patterns"]
        if store_key is not None:
            self.adata.obsm[store_key] = output
        return output

    def get_nn_embed(
        self,
        batch_size: int = 128,
        store_key: str | None = None,
        prog_bar: bool = False,
    ) -> np.ndarray:
        """Return AMICI embedded neighbor expression for the registered AnnData."""
        output = self._collect_outputs(("nn_embed",), batch_size=batch_size, prog_bar=prog_bar)[
            "nn_embed"
        ]
        if store_key is not None:
            self.adata.obsm[store_key] = output
        return output
