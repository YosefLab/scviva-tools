"""Stereoscope models for spatial transcriptomics deconvolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalObsField
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.train._config import merge_kwargs
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

from scviva.external.stereoscope._module import RNADeconv, SpatialDeconv
from scviva.model.base._deconvolution_mixin import SpatialDeconvolutionMixin
from scviva.model.base._spatial_base import SpatialBaseModel

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class RNAStereoscope(UnsupervisedTrainingMixin, SpatialBaseModel):
    """Reimplementation of Stereoscope for the scRNA-seq component :cite:p:`Andersson20`.

    Trains the RNA model whose parameters are then transferred to
    :class:`~scviva.external.SpatialStereoscope` for spatial deconvolution.

    Original implementation: https://github.com/almaan/stereoscope.

    Parameters
    ----------
    sc_adata
        Single-cell AnnData registered via
        :meth:`~scviva.external.RNAStereoscope.setup_anndata`.
    **model_kwargs
        Keyword args for :class:`~scviva.external.stereoscope.RNADeconv`.

    Examples
    --------
    >>> scviva.external.RNAStereoscope.setup_anndata(sc_adata, labels_key="labels")
    >>> sc_model = RNAStereoscope(sc_adata)
    >>> sc_model.train()

    Notes
    -----
    See further usage examples in the following tutorial:

    1. :doc:`/tutorials/stereoscope_heart_LV_tutorial`
    """

    def __init__(self, sc_adata: AnnData, **model_kwargs):
        super().__init__(sc_adata)
        self.n_genes = self.summary_stats.n_vars
        self.n_labels = self.summary_stats.n_labels
        self.module = RNADeconv(
            n_genes=self.n_genes,
            n_labels=self.n_labels,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"RNADeconv Model with params: \nn_genes: {self.n_genes}, n_labels: {self.n_labels}"
        )
        self.init_params_ = self._get_init_params(locals())

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 400,
        lr: float = 0.01,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 1,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Train the model using MAP inference.

        Parameters
        ----------
        max_epochs
            Number of epochs to train for.
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of the training set in [0.0, 1.0].
        validation_size
            Size of the test set.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size.
        datasplitter_kwargs
            Additional kwargs for :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        plan_kwargs = merge_kwargs(None, plan_kwargs, name="plan")
        plan_kwargs.update({"lr": lr})
        super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str | None = None,
        layer: str | None = None,
        batch_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_batch_key)s
        **kwargs
            Keyword args for :meth:`~scvi.data.AnnDataManager.register_fields`.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


class SpatialStereoscope(SpatialDeconvolutionMixin, UnsupervisedTrainingMixin, SpatialBaseModel):
    """Reimplementation of Stereoscope for the spatial component :cite:p:`Andersson20`.

    Deconvolves spatial transcriptomics spots into cell type proportions using
    parameters learned by a pre-trained :class:`~scviva.external.RNAStereoscope` model.

    Inherits :class:`~scviva.model.base.SpatialDeconvolutionMixin` which provides
    :meth:`get_proportions_df` and :meth:`plot_cell_type_map`.

    Parameters
    ----------
    st_adata
        Spatial AnnData registered via
        :meth:`~scviva.external.SpatialStereoscope.setup_anndata`.
    sc_params
        Parameters from the RNA model (from :meth:`~scviva.external.RNAStereoscope.get_params`).
    cell_type_mapping
        numpy array mapping for the cell types used in the deconvolution.
    prior_weight
        How to reweight minibatches. ``"n_obs"`` is statistically correct;
        ``"minibatch"`` reproduces the original Stereoscope paper.
    **model_kwargs
        Keyword args for :class:`~scviva.external.stereoscope.SpatialDeconv`.

    Examples
    --------
    >>> RNAStereoscope.setup_anndata(sc_adata, labels_key="labels")
    >>> sc_model = RNAStereoscope(sc_adata)
    >>> sc_model.train()
    >>> SpatialStereoscope.setup_anndata(st_adata)
    >>> st_model = SpatialStereoscope.from_rna_model(st_adata, sc_model)
    >>> st_model.train()
    >>> st_adata.obsm["deconv"] = st_model.get_proportions()

    Notes
    -----
    See further usage examples in the following tutorial:

    1. :doc:`/tutorials/stereoscope_heart_LV_tutorial`
    """

    def __init__(
        self,
        st_adata: AnnData,
        sc_params: tuple[np.ndarray],
        cell_type_mapping: np.ndarray,
        prior_weight: Literal["n_obs", "minibatch"] = "n_obs",
        **model_kwargs,
    ):
        super().__init__(st_adata)
        self.module = SpatialDeconv(
            n_spots=st_adata.n_obs,
            sc_params=sc_params,
            prior_weight=prior_weight,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"SpatialDeconv Model with params: \nn_spots: {st_adata.n_obs}"
        )
        self.cell_type_mapping = cell_type_mapping
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def from_rna_model(
        cls,
        st_adata: AnnData,
        sc_model: RNAStereoscope,
        prior_weight: Literal["n_obs", "minibatch"] = "n_obs",
        **model_kwargs,
    ):
        """Alternate constructor using a pre-trained RNA model.

        Parameters
        ----------
        st_adata
            Registered spatial AnnData.
        sc_model
            Trained :class:`~scviva.external.RNAStereoscope` model.
        prior_weight
            How to reweight minibatches for stochastic optimization.
        **model_kwargs
            Keyword args for :class:`~scviva.external.stereoscope.SpatialDeconv`.
        """
        return cls(
            st_adata,
            sc_model.module.get_params(),
            sc_model.adata_manager.get_state_registry(
                REGISTRY_KEYS.LABELS_KEY
            ).categorical_mapping,
            prior_weight=prior_weight,
            **model_kwargs,
        )

    def get_proportions(self, keep_noise: bool = False) -> pd.DataFrame:
        """Return the estimated cell type proportions for the spatial data.

        Shape is (n_spots, n_labels) or (n_spots, n_labels + 1) if keep_noise.

        Parameters
        ----------
        keep_noise
            Whether to include the noise term as a standalone cell type.
        """
        self._check_if_trained()
        column_names = self.cell_type_mapping
        if keep_noise:
            column_names = np.append(column_names, "noise_term")
        return pd.DataFrame(
            data=self.module.get_proportions(keep_noise),
            columns=column_names,
            index=self.adata.obs.index,
        )

    def get_scale_for_ct(self, y: np.ndarray) -> np.ndarray:
        r"""Calculate the cell-type-specific expression.

        Parameters
        ----------
        y
            Array of cell type names to query.

        Returns
        -------
        Gene expression array of shape (n_query, n_genes).
        """
        self._check_if_trained()
        ind_y = np.array([np.where(ct == self.cell_type_mapping)[0][0] for ct in y])
        if ind_y.shape != y.shape:
            raise ValueError(
                "Incorrect shape after matching cell types to reference mapping. "
                "Please check cell type query."
            )
        px_scale = self.module.get_ct_specific_expression(torch.tensor(ind_y)[:, None])
        return np.array(px_scale.cpu())

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 400,
        lr: float = 0.01,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Train the model using MAP inference.

        Parameters
        ----------
        max_epochs
            Number of epochs to train for.
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size.
        datasplitter_kwargs
            Additional kwargs for :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        plan_kwargs = merge_kwargs(None, plan_kwargs, name="plan")
        plan_kwargs.update({"lr": lr})
        super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=1,
            validation_size=None,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        **kwargs
            Keyword args for :meth:`~scvi.data.AnnDataManager.register_fields`.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
