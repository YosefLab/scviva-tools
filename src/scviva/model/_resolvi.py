from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import Trace_ELBO
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import AnnTorchDataset
from scvi.model._utils import (
    scrna_raw_counts_properties,
)
from scvi.model.base import ArchesMixin, BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.model.base._de_core import _de_core
from scvi.train._config import merge_kwargs
from scvi.utils import de_dsp, setup_anndata_dsp, track

from scviva.model.base import SpatialBaseModel, SpatialNeighborhoodMixin, SpatialPredictiveMixin
from scviva.module._resolvae import RESOLVAE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class ResolVI(
    SpatialNeighborhoodMixin,
    SpatialPredictiveMixin,
    SpatialBaseModel,
    PyroSviTrainMixin,
    PyroSampleMixin,
    ArchesMixin,
    BaseModelClass,
):
    """
    ResolVI addresses noise and bias in single-cell resolved spatial transcriptomics data.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scviva.model.ResolVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    **model_kwargs
        Keyword args for :class:`~scviva.module.RESOLVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> ResolVI.setup_anndata(adata, batch_key="batch")
    >>> model = ResolVI(adata)
    >>> model.train()
    >>> adata.obsm["X_resolVI"] = model.get_latent_representation()
    >>> adata.layers["X_normalized_resolVI"] = model.get_normalized_expression()
    """

    _module_cls = RESOLVAE
    _block_parameters = []

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 32,
        n_hidden_encoder: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.05,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        background_ratio=None,
        median_distance=None,
        semisupervised=False,
        mixture_k=50,
        downsample_counts=True,
        **model_kwargs,
    ):
        pyro.clear_param_store()

        super().__init__(adata)
        if semisupervised:
            self._set_indices_and_labels()

        results = self.compute_dataset_dependent_priors()

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_labels = self.summary_stats.n_labels - 1

        if background_ratio is None:
            background_ratio = results["background_ratio"]
        if median_distance is None:
            median_distance = results["median_distance"]
        if downsample_counts:
            downsample_counts_mean = results["mean_log_counts"]
            downsample_counts_std = results["std_log_counts"]
        else:
            downsample_counts_mean = None
            downsample_counts_std = 1.0

        expression_anntorchdata = AnnTorchDataset(
            self.adata_manager,
            getitem_tensors=["X"],
            load_sparse_tensor=True,
        )
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_cats_per_cov=n_cats_per_cov,
            n_labels=n_labels,
            mixture_k=mixture_k,
            expression_anntorchdata=expression_anntorchdata,
            n_neighbors=self.summary_stats.n_distance_neighbor,
            n_obs=self.summary_stats["n_ind_x"],
            n_hidden=n_hidden,
            n_hidden_encoder=n_hidden_encoder,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            background_ratio=background_ratio,
            median_distance=median_distance,
            semisupervised=semisupervised,
            downsample_counts_mean=downsample_counts_mean,
            downsample_counts_std=downsample_counts_std,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"ResolVI Model with the following params: \nn_hidden: {n_hidden} "
            f"n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: "
            f"{dropout_rate}, dispersion: {dispersion}, gene_likelihood: {gene_likelihood} "
            f"n_neighbors: {self.summary_stats.n_distance_neighbor}"
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------ #
    # Pyro-specific overrides of SpatialPredictiveMixin defaults
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 1,
        batch_size: int | None = None,
        return_dist: bool = False,
        backend: str = "cpu",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the latent representation for each cell (Pyro path).

        Parameters
        ----------
        adata
            AnnData object. If ``None``, defaults to the model's registered adata.
        indices
            Indices of cells in adata to use. If ``None``, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            Ignored (kept for API consistency with scVI).
        batch_size
            Minibatch size. Defaults to ``scvi.settings.batch_size``.
        return_dist
            Return distribution parameters instead of sampled values.

        Returns
        -------
        Latent representation or ``(mean, variance)`` tuple if ``return_dist=True``.
        """
        import torch as _torch
        from scvi.model._utils import parse_device_args as _parse_device_args

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        latent_qzm = []
        latent_qzv = []

        _, _, device = _parse_device_args(
            accelerator="auto",
            devices="auto",
            return_device="torch",
            validate_single_device=True,
        )

        for tensors in scdl:
            _, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}

            if kwargs["cat_covs"] is not None and self.module.encode_covariates:
                categorical_input = list(_torch.split(kwargs["cat_covs"], 1, dim=1))
            else:
                categorical_input = ()

            qz_m, qz_v, z = self.module.z_encoder(
                _torch.log1p(kwargs["x"] / _torch.mean(kwargs["x"], dim=1, keepdim=True)),
                kwargs["batch_index"],
                *categorical_input,
            )
            qz = _torch.distributions.Normal(qz_m, qz_v.sqrt())
            if give_mean:
                z = qz.loc

            latent += [z.cpu()]
            latent_qzm += [qz.loc.cpu()]
            latent_qzv += [qz.scale.square().cpu()]
        result = (
            (_torch.cat(latent_qzm).numpy(), _torch.cat(latent_qzv).numpy())
            if return_dist
            else _torch.cat(latent).numpy()
        )
        return self._maybe_rapids(result, backend)

    @de_dsp.dedent
    @torch.inference_mode()
    def get_normalized_expression_importance(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[int | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | None = 1,
        n_samples: int = 30,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        weights: str | np.ndarray | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        library_scaling: bool = False,
        size_scaling: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns importance-sampled normalized gene expression (Pyro path).

        Parameters
        ----------
        adata
            AnnData object. If ``None``, defaults to the model's registered adata.
        indices
            Indices of cells to use.
        transform_batch
            Not supported. Kept for API consistency.
        gene_list
            Subset of genes to return.
        library_size
            Scale expression to this common library size.
        n_samples
            Number of posterior samples.
        n_samples_overall
            Overrides ``n_samples``; returns a flat sample array.
        batch_size
            Minibatch size.
        weights
            Precomputed importance weights. ``"uniform"`` disables importance sampling.
        return_mean
            Return the mean over samples.
        return_numpy
            Return :class:`~numpy.ndarray` instead of :class:`~pandas.DataFrame`.
        library_scaling
            Multiply decoded expression by library size.
        size_scaling
            Divide by size factor (requires ``size_factor_key`` in ``setup_anndata``).

        Returns
        -------
        Normalized expression array or DataFrame of shape ``(n_cells, n_genes)``.
        """
        import warnings as _warnings

        from pyro import infer as _infer
        from scvi.model._utils import parse_device_args

        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if transform_batch is not None:
            warnings.warn(
                "`transform_batch` is not supported by "
                "`get_normalized_expression_importance` and will be ignored.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                _warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is `False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        exprs = []
        weighting = []

        _, _, device = parse_device_args(
            accelerator="auto",
            devices="auto",
            return_device="torch",
            validate_single_device=True,
        )

        for tensors in scdl:
            args, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}
            model_now = partial(self.module.model_simplified, corrected_rate=True)
            importance_dist = _infer.Importance(
                model_now, guide=self.module.guide.guide_simplified, num_samples=10 * n_samples
            )
            posterior = importance_dist.run(*args, **kwargs)
            marginal = _infer.EmpiricalMarginal(posterior, sites=["mean_poisson", "px_scale"])
            samples = torch.cat([marginal().unsqueeze(1) for i in range(n_samples)], 1)
            log_weights = (
                torch.distributions.Poisson(samples[0, ...] + 1e-3)
                .log_prob(kwargs["x"].to(samples.device))
                .sum(-1)
            )
            log_weights = log_weights / kwargs["x"].to(samples.device).sum(-1)
            weighting.append(log_weights.reshape(-1).cpu())
            if library_scaling or size_scaling:
                if size_scaling:
                    if "size_factor" in self.adata_manager.data_registry:
                        size_factor = kwargs["size_factor"]
                        samples[0, ...] = samples[0, ...] / size_factor.unsqueeze(0).repeat(
                            n_samples, 1, 1
                        )
                    else:
                        raise ValueError(
                            "size_scaling is True but no size_factor_key was provided "
                            "in setup_anndata."
                        )
                exprs.append(samples[0, ...].cpu())
            else:
                exprs.append(samples[1, ...].cpu())
        exprs = torch.cat(exprs, axis=1).numpy()
        exprs = exprs[..., gene_mask]
        if return_mean:
            exprs = exprs.mean(0)
        weighting = torch.cat(weighting, axis=0).numpy()
        if library_size is not None:
            exprs = library_size * exprs

        if n_samples_overall is not None:
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            if weights == "uniform":
                p = None
            else:
                weighting -= weighting.max()
                weighting = np.exp(weighting)
                p = weighting / weighting.sum(axis=0, keepdims=True)

            ind_ = np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)
            exprs = exprs[ind_]

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs

    @de_dsp.dedent
    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[int | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | None = 1,
        size_scaling: bool = False,
        n_samples: int = 1,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        silent: bool = True,
        **kwargs,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns normalized gene expression (Pyro path).

        Parameters
        ----------
        adata
            AnnData object. If ``None``, defaults to the model's registered adata.
        indices
            Indices of cells to use.
        transform_batch
            Batch to condition on.
        gene_list
            Subset of genes to return.
        library_size
            Scale expression to this common library size.
        size_scaling
            Divide by size factor (requires ``size_factor_key`` in ``setup_anndata``).
        n_samples
            Number of posterior samples.
        n_samples_overall
            Overrides ``n_samples``.
        batch_size
            Minibatch size.
        return_mean
            Return the mean over samples.
        return_numpy
            Return :class:`~numpy.ndarray` instead of :class:`~pandas.DataFrame`.
        silent
            Suppress progress bar.

        Returns
        -------
        Normalized expression array or DataFrame of shape ``(n_cells, n_genes)``.
        """
        import warnings as _warnings

        from scvi.model._utils import _get_batch_code_from_category, parse_device_args

        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                _warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is `False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        exprs = []

        _, _, device = parse_device_args(
            accelerator="auto",
            devices="auto",
            return_device="torch",
            validate_single_device=True,
        )

        for tensors in scdl:
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                _, kw = self.module._get_fn_args_from_batch(tensors)
                kw = {k: v.to(device) if v is not None else v for k, v in kw.items()}

                if kw["cat_covs"] is not None and self.module.encode_covariates:
                    categorical_input = list(torch.split(kw["cat_covs"], 1, dim=1))
                else:
                    categorical_input = ()

                qz_m, qz_v, _ = self.module.z_encoder(
                    torch.log1p(kw["x"] / torch.mean(kw["x"], dim=1, keepdim=True)),
                    kw["batch_index"],
                    *categorical_input,
                )
                z = torch.distributions.Normal(qz_m, qz_v.sqrt()).sample([n_samples])

                if kw["cat_covs"] is not None:
                    categorical_input = list(torch.split(kw["cat_covs"], 1, dim=1))
                else:
                    categorical_input = ()
                if batch is not None:
                    batch = torch.full_like(kw["batch_index"], batch)
                else:
                    batch = kw["batch_index"]

                px_scale, _, px_rate, _ = self.module.model.decoder(
                    self.module.model.dispersion, z, kw["library"], batch, *categorical_input
                )
                if size_scaling:
                    if "size_factor" in self.adata_manager.data_registry:
                        size_factor = kw["size_factor"]
                        px_rate = px_rate / size_factor.reshape(-1, 1, 1)
                        exp_ = px_rate
                    else:
                        raise ValueError(
                            "size_scaling is True but no size_factor_key was provided "
                            "in setup_anndata."
                        )
                else:
                    exp_ = library_size * px_scale if library_size is not None else px_rate

                exp_ = exp_[..., gene_mask]
                per_batch_exprs.append(exp_[None].cpu())
            per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
            exprs.append(per_batch_exprs)

        exprs = np.concatenate(exprs, axis=1)
        if return_mean:
            exprs = exprs.mean(0)

        if n_samples_overall is not None:
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            ind_ = np.random.choice(n_samples_, n_samples_overall, replace=True)
            exprs = exprs[ind_]

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs

    def train(
        self,
        max_epochs: int = 50,
        lr: float = 3e-3,
        lr_extra: float = 1e-2,
        extra_lr_parameters: tuple = ("per_neighbor_diffusion_map", "u_prior_means"),
        batch_size: int = 512,
        weight_decay: float = 0.0,
        eps: float = 1e-4,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 20,
        plan_kwargs: dict | None = None,
        expose_params: list = (),
        **kwargs,
    ):
        """
        Train the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        lr_extra
            Learning rate for parameters (non-amortized and custom ones)
        extra_lr_parameters
            List of parameters to train with `lr_extra` learning rate.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        plan_kwargs
            Keyword args for the Pyro training plan.
        expose_params
            List of parameters to train if running model in Arches mode.
        **kwargs
            Other keyword args for the Trainer.

        Notes
        -----
        RESOLVI trains with Pyro SVI and maintains per-cell global parameters, so it does not
        support a held-out validation set. ``train_size`` must be ``1.0`` and ``early_stopping``
        is not available.
        """
        train_size = kwargs.pop("train_size", 1.0)
        if train_size != 1.0:
            raise ValueError(
                "RESOLVI does not support a validation set: it uses Pyro SVI with per-cell "
                f"global parameters, so `train_size` must be 1.0 (got {train_size})."
            )
        if kwargs.pop("early_stopping", False):
            raise ValueError(
                "RESOLVI does not support `early_stopping` because it trains without a "
                "validation set (`train_size` must be 1.0)."
            )

        blocked = self._block_parameters.copy()
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                blocked.append(name)
                param.requires_grad = True
        blocked = set(blocked) - set(expose_params)

        if blocked:
            print("Running transfer learning mode. Set lr to 0 and blocking variables.")

        def per_param_callable(module_name, param_name):
            store_name = f"{module_name}$$${param_name}" if "." in param_name else param_name
            if store_name in blocked:
                return {"lr": 0.0, "weight_decay": 0, "eps": eps}
            if store_name in extra_lr_parameters:
                return {"lr": lr_extra, "weight_decay": weight_decay, "eps": eps}
            else:
                return {"lr": lr, "weight_decay": weight_decay, "eps": eps}

        optim = pyro.optim.Adam(per_param_callable)

        plan_kwargs = merge_kwargs(None, plan_kwargs, name="plan")
        plan_kwargs.update(
            {
                "optim_kwargs": {"lr": lr, "weight_decay": weight_decay, "eps": eps},
                "optim": optim,
                "blocked": blocked,
                "n_epochs_kl_warmup": n_epochs_kl_warmup
                if n_epochs_kl_warmup is not None
                else max_epochs,
                "n_steps_kl_warmup": n_steps_kl_warmup,
                "loss_fn": Trace_ELBO(
                    num_particles=5, vectorize_particles=True, retain_graph=True
                ),
            }
        )

        super().train(
            max_epochs=max_epochs,
            train_size=1.0,
            plan_kwargs=plan_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    @staticmethod
    def _prepare_data(
        adata: AnnData,
        n_neighbors: int = 10,
        spatial_rep: str = "X_spatial",
        batch_key: str | None = None,
        slice_key: str | None = None,
        **kwargs,
    ) -> None:
        """Compute spatial neighbors and store in ``adata.obsm``.

        Parameters
        ----------
        adata
            AnnData object with spatial coordinates.
        n_neighbors
            Number of spatial neighbors to compute.
        spatial_rep
            Key in ``adata.obsm`` containing spatial coordinates.
        batch_key
            Key in ``adata.obs`` for batch/slice grouping of neighbor computation.
        slice_key
            Alias for ``batch_key``.
        """
        if slice_key is not None:
            batch_key = slice_key
        try:
            import scanpy
            from sklearn.neighbors._base import _kneighbors_from_graph
        except ImportError as err:
            raise ImportError(
                "Please install scanpy and scikit-learn -- `pip install scanpy`"
            ) from err

        if batch_key is None:
            indices = [np.arange(adata.n_obs)]
        else:
            indices = [
                np.where(adata.obs[batch_key] == i)[0] for i in adata.obs[batch_key].unique()
            ]

        distance_neighbor = 1e6 * np.ones([adata.n_obs, n_neighbors])
        index_neighbor = np.zeros([adata.n_obs, n_neighbors], dtype=int)

        for index in indices:
            sub_data = adata[index].copy()
            try:
                import rapids_singlecell

                rapids_singlecell.pp.neighbors(
                    sub_data, n_neighbors=n_neighbors + 5, use_rep=spatial_rep
                )
            except ImportError:
                scanpy.pp.neighbors(sub_data, n_neighbors=n_neighbors + 5, use_rep=spatial_rep)
            distances = sub_data.obsp["distances"] ** 2

            distance_neighbor[index, :], index_neighbor_batch = _kneighbors_from_graph(
                distances, n_neighbors, return_distance=True
            )
            index_neighbor[index, :] = index[index_neighbor_batch]

        adata.obsm["X_spatial"] = adata.obsm[spatial_rep]
        adata.obsm["index_neighbor"] = index_neighbor
        adata.obsm["distance_neighbor"] = distance_neighbor

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        prepare_data: bool | None = True,
        prepare_data_kwargs: dict | None = None,
        unlabeled_category: str = "unknown",
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        size_factor_key
            Key in ``adata.obs`` corresponding to pre-computed size factors.
        %(param_cat_cov_keys)s
        prepare_data
            If ``True``, automatically compute spatial neighbors via :meth:`_prepare_data`.
            Set to ``False`` if neighbors are already in ``adata.obsm``.
        prepare_data_kwargs
            Keyword args for :meth:`_prepare_data` (e.g. ``n_neighbors``, ``spatial_rep``).
        %(param_unlabeled_category)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        x = adata.X if layer is None else adata.layers[layer]
        assert np.min(x.sum(axis=1)) > 0, (
            "Please filter cells with less than 5 counts prior to running ResolVI."
        )
        if prepare_data:
            if prepare_data_kwargs is None:
                prepare_data_kwargs = {}
            spatial_rep = prepare_data_kwargs.get("spatial_rep", "X_spatial")
            effective_config = {"batch_key": batch_key, **prepare_data_kwargs}
            stored_config = adata.uns.get("_resolvi_prepare_data_config", None)
            neighbors_valid = (
                stored_config == effective_config
                and "index_neighbor" in adata.obsm
                and "distance_neighbor" in adata.obsm
                and int(adata.obsm["index_neighbor"].max()) < adata.n_obs
            )
            if not neighbors_valid and spatial_rep in adata.obsm:
                print(
                    "Preparing data for training. This may take a while. "
                    "RAPIDS SingleCell will be used if installed."
                )
                cls._prepare_data(adata, batch_key=batch_key, **prepare_data_kwargs)
                adata.uns["_resolvi_prepare_data_config"] = effective_config
            elif "index_neighbor" not in adata.obsm:
                raise KeyError(
                    f"Spatial key '{spatial_rep}' not found in adata.obsm and no pre-computed "
                    "neighbors found. Either provide spatial coordinates or pre-compute "
                    "neighbors manually and call setup_anndata with prepare_data=False."
                )
        if batch_key is not None:
            adata.obs["_indices"] = (
                adata.obs[batch_key].astype(str) + "_" + adata.obs_names.astype(str)
            )
        else:
            adata.obs["_indices"] = adata.obs_names.astype(str)
        adata.obs["_indices"] = adata.obs["_indices"].astype("category")
        assert not adata.obs["_indices"].duplicated(keep="first").any(), (
            "obs_names need to be unique prior to running ResolVI."
        )
        if labels_key is None:
            label_field = CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key)
        else:
            label_field = LabelsWithUnlabeledObsField(
                REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category
            )

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            ObsmField("index_neighbor", "index_neighbor"),
            ObsmField("distance_neighbor", "distance_neighbor"),
            CategoricalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
            label_field,
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def compute_dataset_dependent_priors(self, n_small_genes=None):
        """Compute dataset-dependent priors for background ratio and median distance."""
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        n_small_genes = x.shape[1] // 50 if n_small_genes is None else int(n_small_genes)
        smallest_means = x[:, np.array(x.sum(0)).flatten().argsort()[:n_small_genes]].mean(
            1
        ) / np.array(x.mean(1))
        # Cast off numpy's float64 scalars: RESOLVAE registers these as buffers via
        # torch.tensor(...), which otherwise inherits float64 and can't move to MPS
        # (no float64 kernel support there), matching the float() idiom already used
        # for downsample_counts_mean/std in RESOLVAE.
        background_ratio = float(np.mean(np.array(smallest_means)))

        distance = self.adata_manager.get_from_registry("distance_neighbor")
        median_distance = float(np.median(np.partition(distance, 5)[:, 5]))
        log_library_size = np.log1p(np.array(x.sum(1)))
        mean_log_counts = float(np.median(log_library_size))
        std_log_counts = float(np.std(log_library_size))

        return {
            "background_ratio": background_ratio,
            "median_distance": median_distance,
            "mean_log_counts": mean_log_counts,
            "std_log_counts": std_log_counts,
        }

    @de_dsp.dedent
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        subset_idx: Sequence[int] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        weights: Literal["uniform", "importance"] | None = "uniform",
        filter_outlier_cells: bool = False,
        n_samples: int = 5,
        size_scaling: bool = False,
        library_scaling: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential expression analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18` and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_subset_idx)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        weights
            Precomputed weight for importance sampling.
        filter_outlier_cells
            Whether to filter outlier cells.
        n_samples
            Number of posterior samples to use for estimation.
        size_scaling
            If True, will scale normalized expression by size factors.
        library_scaling
            If True, will scale normalized expression to library size.
        **kwargs
            Keyword args for DifferentialComputation.get_bayes_factors

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        if library_scaling and weights != "importance":
            raise ValueError(
                "library_scaling=True is only supported with weights='importance'. "
                "Pass weights='importance' or set library_scaling=False."
            )

        if weights == "importance":
            model_fn = partial(
                self.get_normalized_expression_importance,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                weights=weights,
                return_mean=False,
                size_scaling=size_scaling,
                library_scaling=library_scaling,
            )
        else:
            model_fn = partial(
                self.get_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                return_mean=False,
                size_scaling=size_scaling,
            )

        representation_fn = self.get_latent_representation if filter_outlier_cells else None

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=representation_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            subset_idx=subset_idx,
            all_stats=all_stats,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=adata.var_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )

        return result

    @de_dsp.dedent
    def differential_niche_abundance(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        neighbor_key: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        subset_idx: Sequence[int] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        filter_outlier_cells: bool = False,
        pseudocounts: float = 1e-3,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for niche differential abundance analysis.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        neighbor_key
            Obsm key containing the spatial neighbors of each cell.
        %(de_idx1)s
        %(de_idx2)s
        %(de_subset_idx)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_fdr_target)s
        %(de_silent)s
        filter_outlier_cells
            Whether to filter outlier cells.
        pseudocounts
            pseudocount offset used for the mode `change`.
        **kwargs
            Keyword args for DifferentialComputation.get_bayes_factors

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        model_fn = partial(
            self.get_neighbor_abundance,
            return_numpy=True,
            n_samples=5,
            batch_size=batch_size,
            return_mean=False,
            neighbor_key=neighbor_key,
        )

        representation_fn = self.get_latent_representation if filter_outlier_cells else None

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=representation_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            subset_idx=subset_idx,
            all_stats=False,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=self._label_mapping[:-1],
            mode=mode,
            batchid1=None,
            batchid2=None,
            delta=delta,
            batch_correction=False,
            fdr=fdr_target,
            silent=silent,
            pseudocounts=pseudocounts,
            **kwargs,
        )

        return result

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        soft: bool = False,
        batch_size: int | None = 500,
        num_samples: int | None = 30,
    ) -> np.ndarray | pd.DataFrame:
        """
        Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via setup_anndata.
        indices
            Subsample AnnData to these indices.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into the model.
        num_samples
            Samples to draw from the posterior for cell-type prediction.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        sampled_prediction = self.sample_posterior(
            adata=adata,
            indices=indices,
            model=self.module.model_corrected,
            return_sites=["probs_prediction"],
            num_samples=num_samples,
            return_samples=False,
            batch_size=batch_size,
            summary_frequency=10,
            return_observed=True,
        )
        y_pred = sampled_prediction["post_sample_means"]["probs_prediction"]

        if not soft:
            y_pred = y_pred.argmax(axis=1)
            predictions = [self._code_to_label[p] for p in y_pred]
            return np.array(predictions)
        else:
            n_labels = len(y_pred[0])
            predictions = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
            return predictions

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells."""
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))
