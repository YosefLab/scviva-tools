"""GIMVI model for imputing missing genes across spatial and scRNA-seq data."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._compat import registry_from_setup_dict
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _init_library_size, parse_device_args
from scvi.train import Trainer
from scvi.train._config import merge_kwargs
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

from scviva.model.base._spatial_base import SpatialBaseModel
from scviva.model.utils._dataloaders import CyclicMultiDataLoader as TrainDL
from scviva.model.utils._gimvi_utils import (
    _load_legacy_saved_gimvi_files,
    _load_saved_gimvi_files,
)
from scviva.module._jvae import JVAE
from scviva.train._gimvi_trainingplans import GIMVITrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData
    from scvi.dataloaders import AnnDataLoader

logger = logging.getLogger(__name__)


def _unpack_tensors(tensors):
    x = tensors[REGISTRY_KEYS.X_KEY].squeeze_(0)
    batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze_(0)
    y = tensors[REGISTRY_KEYS.LABELS_KEY].squeeze_(0)
    return x, batch_index, y


class GIMVI(SpatialBaseModel):
    """Joint VAE for imputing missing genes in spatial data :cite:p:`Lopez19`.

    Learns a joint latent space for paired scRNA-seq and spatial transcriptomics
    data, enabling imputation of spatially unmeasured genes.

    Parameters
    ----------
    adata_seq
        AnnData object registered via :meth:`~scviva.model.GIMVI.setup_anndata`
        containing scRNA-seq data.
    adata_spatial
        AnnData object registered via :meth:`~scviva.model.GIMVI.setup_anndata`
        containing spatial transcriptomics data.
    generative_distributions
        List of generative distributions for seq and spatial data.
        Defaults to ``['zinb', 'nb']``.
    model_library_size
        Whether to model library size per dataset. Defaults to ``[True, False]``.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~scviva.module.JVAE`.

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_seq)
    >>> adata_spatial = anndata.read_h5ad(path_to_spatial)
    >>> scviva.model.GIMVI.setup_anndata(adata_seq)
    >>> scviva.model.GIMVI.setup_anndata(adata_spatial)
    >>> model = scviva.model.GIMVI(adata_seq, adata_spatial)
    >>> model.train(max_epochs=200)

    Notes
    -----
    See further usage examples in the following tutorial:

    1. :doc:`/tutorials/gimvi_tutorial`
    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        generative_distributions: list[str] | None = None,
        model_library_size: list[bool] | None = None,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super().__init__(adata_seq)
        if adata_seq is adata_spatial:
            raise ValueError(
                "`adata_seq` and `adata_spatial` cannot point to the same object. "
                "If you would really like to do this, make a copy and pass it as `adata_spatial`."
            )
        model_library_size = model_library_size or [True, False]
        generative_distributions = generative_distributions or ["zinb", "nb"]
        self.adatas = [adata_seq, adata_spatial]
        self.adata_managers = {
            "seq": self._get_most_recent_anndata_manager(adata_seq, required=True),
            "spatial": self._get_most_recent_anndata_manager(adata_spatial, required=True),
        }
        self.registries_ = []
        for adm in self.adata_managers.values():
            self._register_manager_for_instance(adm)
            self.registries_.append(adm.registry)

        seq_var_names = adata_seq.var_names
        spatial_var_names = adata_spatial.var_names

        if not set(spatial_var_names) <= set(seq_var_names):
            raise ValueError("spatial genes must be a subset of seq genes")

        spatial_gene_loc = [np.argwhere(seq_var_names == g)[0] for g in spatial_var_names]
        spatial_gene_loc = np.concatenate(spatial_gene_loc)
        gene_mappings = [slice(None), spatial_gene_loc]
        sum_stats = [adm.summary_stats for adm in self.adata_managers.values()]
        n_inputs = [s["n_vars"] for s in sum_stats]
        total_genes = n_inputs[0]

        adata_seq_n_batches = sum_stats[0]["n_batch"]
        adata_spatial_batch = adata_spatial.obs[
            self.adata_managers["spatial"].data_registry[REGISTRY_KEYS.BATCH_KEY].attr_key
        ]
        if np.min(adata_spatial_batch) == 0:
            adata_spatial.obs[
                self.adata_managers["spatial"].data_registry[REGISTRY_KEYS.BATCH_KEY].attr_key
            ] += adata_seq_n_batches

        n_batches = sum(s["n_batch"] for s in sum_stats)

        library_log_means = []
        library_log_vars = []
        for adata_manager in self.adata_managers.values():
            adata_library_log_means, adata_library_log_vars = _init_library_size(
                adata_manager, n_batches
            )
            library_log_means.append(adata_library_log_means)
            library_log_vars.append(adata_library_log_vars)

        self.module = JVAE(
            n_inputs,
            total_genes,
            gene_mappings,
            generative_distributions,
            model_library_size,
            library_log_means,
            library_log_vars,
            n_batch=n_batches,
            n_latent=n_latent,
            **model_kwargs,
        )

        self._model_summary_string = (
            "GIMVI Model with the following params: \n"
            f"n_latent: {n_latent}, n_inputs: {n_inputs}, n_genes: {total_genes}, "
            f"n_batch: {n_batches}, generative distributions: {generative_distributions}"
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------ #
    # SpatialData integration (override to clarify spatial-adata role)
    # ------------------------------------------------------------------ #

    @classmethod
    def setup_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **kwargs,
    ) -> None:
        """Register the *spatial* adata component from a SpatialData object.

        Calls :meth:`setup_anndata` on the extracted table. The scRNA-seq
        component must always be registered via :meth:`setup_anndata` directly.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the spatial AnnData table.
        region
            Region name to subset (stored in ``sdata[table_key].obs``).
        **kwargs
            Passed to :meth:`setup_anndata`.
        """
        super().setup_spatialdata(sdata, table_key=table_key, region=region, **kwargs)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 200,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        kappa: int = 5,
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        %(param_accelerator)s
        %(param_devices)s
        kappa
            Scaling parameter for the discriminator loss.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        batch_size
            Minibatch size.
        datasplitter_kwargs
            Additional kwargs for :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for the training plan.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        accelerator, devices, device = parse_device_args(
            accelerator=accelerator,
            devices=devices,
            return_device="torch",
        )
        datasplitter_kwargs = datasplitter_kwargs or {}

        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **kwargs,
        )
        self.train_indices_, self.test_indices_, self.validation_indices_ = [], [], []
        train_dls, test_dls, val_dls = [], [], []
        for i, adm in enumerate(self.adata_managers.values()):
            ds = DataSplitter(
                adm,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                shuffle_set_split=shuffle_set_split,
                **datasplitter_kwargs,
            )
            ds.setup()
            train_dls.append(ds.train_dataloader())
            test_dls.append(ds.test_dataloader())
            val = ds.val_dataloader()
            val_dls.append(val)
            val.mode = i
            self.train_indices_.append(ds.train_idx)
            self.test_indices_.append(ds.test_idx)
            self.validation_indices_.append(ds.val_idx)
        train_dl = TrainDL(train_dls)

        plan_kwargs = merge_kwargs(None, plan_kwargs, name="plan")
        self._training_plan = GIMVITrainingPlan(
            self.module,
            adversarial_classifier=True,
            scale_adversarial_loss=kappa,
            **plan_kwargs,
        )

        if train_size == 1.0:
            self.trainer.fit(self._training_plan, train_dl)
        else:
            self.trainer.fit(self._training_plan, train_dl, val_dls)
        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()
        self.to_device(device)
        self.is_trained_ = True

    # ------------------------------------------------------------------ #
    # Posterior queries
    # ------------------------------------------------------------------ #

    def _make_scvi_dls(
        self, adatas: list[AnnData] = None, batch_size: int = 128
    ) -> list[AnnDataLoader]:
        if adatas is None:
            adatas = self.adatas
        post_list = [self._make_data_loader(ad, batch_size=batch_size) for ad in adatas]
        for i, dl in enumerate(post_list):
            dl.mode = i
        return post_list

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adatas: list[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
        backend: str = "cpu",
    ) -> list[np.ndarray]:
        """Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of [adata_seq, adata_spatial]. If None, uses the training adatas.
        deterministic
            If True, use the encoder mean instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading.
        backend
            ``"cpu"`` returns numpy arrays.
            ``"rapids"`` transfers each latent array to a cupy array for GPU
            downstream processing (requires ``pip install cuml``).

        Returns
        -------
        List of arrays, one per dataset (seq then spatial).
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latents = []
        for mode, scdl in enumerate(scdls):
            latent = []
            for tensors in scdl:
                (sample_batch, *_) = _unpack_tensors(tensors)
                latent.append(
                    self.module.sample_from_posterior_z(
                        sample_batch, mode, deterministic=deterministic
                    )
                    .cpu()
                    .detach()
                )
            latent_arr = torch.cat(latent).numpy()
            latents.append(latent_arr)

        if backend == "rapids":
            try:
                import cupy as cp

                return [cp.asarray(z) for z in latents]
            except ImportError as e:
                raise ImportError(
                    "backend='rapids' requires cupy. "
                    "Install with: pip install 'scviva-tools[rapids]'"
                ) from e

        return latents

    @torch.inference_mode()
    def get_imputed_values(
        self,
        adatas: list[AnnData] = None,
        deterministic: bool = True,
        normalized: bool = True,
        decode_mode: int | None = None,
        batch_size: int = 128,
    ) -> list[np.ndarray]:
        """Return imputed values for all genes for each dataset.

        Parameters
        ----------
        adatas
            List of adata_seq and adata_spatial.
        deterministic
            If True, use the encoder mean.
        normalized
            Return normalized values or raw rates.
        decode_mode
            If provided, use the decoder of this dataset id for all inputs.
        batch_size
            Minibatch size.
        """
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        imputed_values = []
        for mode, scdl in enumerate(scdls):
            imputed_value = []
            for tensors in scdl:
                (sample_batch, batch_index, label, *_) = _unpack_tensors(tensors)
                if normalized:
                    imputed_value.append(
                        self.module.sample_scale(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                        .cpu()
                        .detach()
                    )
                else:
                    imputed_value.append(
                        self.module.sample_rate(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                        .cpu()
                        .detach()
                    )
            imputed_value = torch.cat(imputed_value).numpy()
            imputed_values.append(imputed_value)

        return imputed_values

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(
        self,
        dir_path: str,
        prefix: str | None = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        save_kwargs: dict | None = None,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.

        Parameters
        ----------
        dir_path
            Path to a directory.
        prefix
            Prefix to prepend to saved file names.
        overwrite
            Overwrite existing data or not.
        save_anndata
            If True, also saves the anndata objects.
        save_kwargs
            Keyword arguments passed into :func:`~torch.save`.
        **anndata_write_kwargs
            Keyword arguments passed into :meth:`~anndata.AnnData.write` when
            ``save_anndata=True``.
        """
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                f"{dir_path} already exists. Provide a non-existing directory for saving."
            )

        file_name_prefix = prefix or ""
        save_kwargs = save_kwargs or {}

        seq_adata = self.adatas[0]
        spatial_adata = self.adatas[1]
        if save_anndata:
            seq_adata.write(
                os.path.join(dir_path, f"{file_name_prefix}adata_seq.h5ad"),
                **anndata_write_kwargs,
            )
            spatial_adata.write(
                os.path.join(dir_path, f"{file_name_prefix}adata_spatial.h5ad"),
                **anndata_write_kwargs,
            )

        model_state_dict = self.module.state_dict()
        seq_var_names = seq_adata.var_names.astype(str).to_numpy()
        spatial_var_names = spatial_adata.var_names.astype(str).to_numpy()

        user_attributes = self._get_user_attributes()
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "seq_var_names": seq_var_names,
                "spatial_var_names": spatial_var_names,
                "attr_dict": user_attributes,
            },
            os.path.join(dir_path, f"{file_name_prefix}model.pt"),
            **save_kwargs,
        )

    @classmethod
    def convert_legacy_save(
        cls,
        dir_path: str,
        output_dir_path: str,
        overwrite: bool = False,
        prefix: str | None = None,
        **save_kwargs,
    ) -> None:
        """Convert a legacy saved GIMVI model (<v0.15.0) to the updated save format.

        Parameters
        ----------
        dir_path
            Path to directory where legacy model is saved.
        output_dir_path
            Path to save converted save files.
        overwrite
            Overwrite existing data or not. If ``False`` and directory
            already exists at ``output_dir_path``, error will be raised.
        prefix
            Prefix of saved file names.
        **save_kwargs
            Keyword arguments passed into :func:`~torch.save`.
        """
        if not os.path.exists(output_dir_path) or overwrite:
            os.makedirs(output_dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                f"{output_dir_path} already exists. Provide a non-existing directory for saving."
            )

        file_name_prefix = prefix or ""
        (
            model_state_dict,
            seq_var_names,
            spatial_var_names,
            attr_dict,
            _,
            _2,
        ) = _load_legacy_saved_gimvi_files(
            dir_path,
            file_name_prefix,
            load_seq_adata=False,
            load_spatial_adata=False,
        )
        if "scvi_setup_dicts_" in attr_dict:
            scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
            registries = []
            for scvi_setup_dict in scvi_setup_dicts:
                registries.append(registry_from_setup_dict(cls, scvi_setup_dict))
            attr_dict["registries_"] = registries

        model_save_path = os.path.join(output_dir_path, f"{file_name_prefix}model.pt")
        torch.save(
            {
                "model_state_dict": model_state_dict,
                "seq_var_names": seq_var_names,
                "spatial_var_names": spatial_var_names,
                "attr_dict": attr_dict,
            },
            model_save_path,
            **save_kwargs,
        )

    @classmethod
    @devices_dsp.dedent
    def load(
        cls,
        dir_path: str,
        adata_seq: AnnData | None = None,
        adata_spatial: AnnData | None = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        prefix: str | None = None,
        backup_url: str | None = None,
    ):
        """Instantiate a model from the saved output.

        Parameters
        ----------
        dir_path
            Path to saved outputs.
        adata_seq
            scRNA-seq AnnData. If None, will check for saved anndata.
        adata_spatial
            Spatial AnnData. If None, will check for saved anndata.
        %(param_accelerator)s
        %(param_device)s
        prefix
            Prefix of saved file names.
        backup_url
            URL to retrieve saved outputs from if not present on disk.

        Returns
        -------
        Model with loaded state dictionaries.
        """
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
        )

        (
            attr_dict,
            seq_var_names,
            spatial_var_names,
            model_state_dict,
            loaded_adata_seq,
            loaded_adata_spatial,
        ) = _load_saved_gimvi_files(
            dir_path,
            adata_seq is None,
            adata_spatial is None,
            prefix=prefix,
            map_location=device,
            backup_url=backup_url,
        )
        adata_seq = loaded_adata_seq or adata_seq
        adata_spatial = loaded_adata_spatial or adata_spatial
        adatas = [adata_seq, adata_spatial]
        var_names = [seq_var_names, spatial_var_names]

        for i, adata in enumerate(adatas):
            saved_var_names = var_names[i]
            user_var_names = adata.var_names.astype(str)
            if not np.array_equal(saved_var_names, user_var_names):
                warnings.warn(
                    "var_names for adata passed in does not match var_names of adata "
                    "used to train the model. For valid results, the vars need to be the "
                    "same and in the same order as the adata used to train the model.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )

        registries = attr_dict.pop("registries_")
        for adata, registry in zip(adatas, registries, strict=True):
            if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
                raise ValueError("It appears you are loading a model from a different class.")
            if _SETUP_ARGS_KEY not in registry:
                raise ValueError(
                    "Saved model does not contain original setup inputs. "
                    "Cannot load the original setup."
                )
            cls.setup_anndata(adata, source_registry=registry, **registry[_SETUP_ARGS_KEY])

        init_params = attr_dict.pop("init_params_")
        if "non_kwargs" in init_params.keys():
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            non_kwargs = {k: v for k, v in init_params.items() if not isinstance(v, dict)}
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        model = cls(adata_seq, adata_spatial, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        model.module.load_state_dict(model_state_dict)
        model.module.eval()
        model.to_device(device)
        return model

    # ------------------------------------------------------------------ #
    # AnnData setup
    # ------------------------------------------------------------------ #

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        layer: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Call once for ``adata_seq`` and once for ``adata_spatial`` before constructing the model.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        **kwargs
            Keyword args for :meth:`~scvi.data.AnnDataManager.register_fields`.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
