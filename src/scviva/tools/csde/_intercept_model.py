import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests
from torch.func import functional_call, grad, hessian, vmap
from tqdm import tqdm

from ._base import PPIAbstractClass
from ._optimization import _zstat_generic2, optimize_ppi_gd, optimize_ppi_lbfgs

_HESSIAN_VMAP_RESIZE_WARNING = ".*was resized since it had shape.*"
_HESSIAN_VMAP_JIT_DEPRECATION_WARNING = ".*torch\\.jit\\.script.*is deprecated"

ParamSpec = list[tuple[str, tuple[int, ...]]]


class InterceptPPIModel(PPIAbstractClass):
    """Shared PPI machinery for per-class-intercept GLMs (Poisson, NB, ...).

    Subclasses provide `self.model` (a `torch.nn.Module` implementing the noise-model-specific
    log-likelihood, with `forward(x, y, w=None) -> {"loss": Tensor, "loss_unsummed": Tensor}`)
    and `self.param_spec` (ordered `(name, shape)` pairs describing that module's parameters;
    exactly one entry is named `"mu"` with shape `(n_classes-1, n_features)` — the per-class
    contrast block; every other entry is a single-per-feature nuisance parameter, e.g. `"mu0"`
    (intercept) or `"rho"` (NB dispersion), contributing a zero contrast row in differential-
    expression testing). Call `self._finalize_init()` at the end of the subclass `__init__`,
    once `self.model`/`self.param_spec` are set.
    """

    def __init__(
        self,
        optimizer: str = "gd",
        optimizer_kwargs: dict[str, Any] | None = None,
        importance_weights: np.ndarray | None = None,
        device: str = "cpu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.device = torch.device(device)

        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl
        all_y = np.hstack([y_gt, y_hat, y_unl])
        unique_y = np.unique(all_y)
        self.n_classes = unique_y.shape[0]
        assert np.isin(np.arange(self.n_classes), unique_y).all()

        self.inputs_gt = (x_gt, y_gt)
        self.inputs_hat = (x_hat, y_hat)
        self.inputs_unl = (x_unl, y_unl)

        if importance_weights is not None:
            if importance_weights.shape != (x_gt.shape[0],):
                raise ValueError(
                    "importance_weights must be a 1-D array with the same length "
                    "as the number of ground-truth observations"
                )
            self.importance_weights = (
                float(x_gt.shape[0]) * importance_weights / importance_weights.sum()
            )
        else:
            self.importance_weights = None

        self.n_features = x_gt.shape[1]
        self.model: torch.nn.Module | None = None
        self.param_spec: ParamSpec = []
        self.model_params = None
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

    def _finalize_init(self):
        """Call at the end of subclass `__init__`, once `self.model`/`self.param_spec` are set."""
        assert self.param_spec[0][0] == "mu", (
            "param_spec[0] must be the 'mu' (per-class contrast) entry"
        )
        assert self.param_spec[0][1] == (self.n_classes - 1, self.n_features), (
            f"'mu' entry must have shape (n_classes-1, n_features) = "
            f"({self.n_classes - 1}, {self.n_features}), got {self.param_spec[0][1]}"
        )
        sizes = [int(np.prod(shape)) for _, shape in self.param_spec]
        self.n_params = int(sum(sizes))
        self._param_sizes = dict(zip((name for name, _ in self.param_spec), sizes, strict=True))
        self.zero_init()
        self.lambd_ = None

    def zero_init(self):
        self.model_params = {
            name: torch.zeros(*shape, dtype=torch.float32, device=self.device)
            for name, shape in self.param_spec
        }

    def _to_tensor(self, arr, dtype=torch.float32):
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=self.device)

    def fit(self, lambd_: float | np.ndarray | None = None, refit: bool = False):
        if lambd_ is None:
            lambd_ = self.get_lambda()
        self.lambd_ = lambd_
        if refit:
            self.zero_init()
        self.theta = self.get_pointestimate(lambd_=lambd_)

    def get_lambda(
        self,
        lambd_0: float = 0.5,
        idx_to_optimize: int | list[int] | None = None,
    ) -> float | np.ndarray:
        self.theta = self.get_pointestimate(lambd_=lambd_0)

        hess = self.hessian_fn(self.inputs_gt, importance_weights=self.importance_weights)
        inv_hess = np.linalg.pinv(hess)
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat, w=self.importance_weights)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt, w=self.importance_weights)

        grad_f_hat_ = grad_f_hat - grad_f_hat.mean(0)
        grad_f_gt_ = grad_f_gt - grad_f_gt.mean(0)
        cov1 = (grad_f_hat_.T @ grad_f_gt_) / self.n
        cov2 = (grad_f_gt_.T @ grad_f_hat_) / self.n

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        num = inv_hess @ (cov1 + cov2) @ inv_hess
        denom = 2 * (1.0 + self.r) * (inv_hess @ vf @ inv_hess)
        if self.lambd_mode == "element":
            lambd_design = [
                np.where(self._construct_contrast(feature_id, 1))[0][0]
                for feature_id in range(self.n_features)
            ]
            lambd_star = np.diag(num / denom)
            return lambd_star[lambd_design]
        if idx_to_optimize is not None:
            if isinstance(idx_to_optimize, int):
                return (
                    num[idx_to_optimize, idx_to_optimize] / denom[idx_to_optimize, idx_to_optimize]
                )
            return np.trace(num[idx_to_optimize, :][:, idx_to_optimize]) / np.trace(
                denom[idx_to_optimize, :][:, idx_to_optimize]
            )
        return np.trace(num) / np.trace(denom)

    def compute_sigma(self, lambd: float | np.ndarray) -> np.ndarray:
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat, w=self.importance_weights)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt, w=self.importance_weights)

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        if self.lambd_mode == "element":
            lambd_good = lambd[self.idx_to_feat()]
        else:
            lambd_good = lambd
        grad_f_ = lambd_good * grad_f_
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        rect_ = grad_f_gt - lambd_good * grad_f_hat
        rect_ = rect_ - rect_.mean(axis=0)
        vdelta = (rect_.T @ rect_) / self.n
        v = vdelta + (self.r * vf)

        hess = self.hessian_fn(self.inputs_gt, importance_weights=self.importance_weights)
        self.hessian = hess
        self.v = v
        return self._compute_sigma(hess, v, self.n)

    def get_pointestimate(self, lambd_: float | np.ndarray) -> np.ndarray:
        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl

        x_gt_t, y_gt_t = self._to_tensor(x_gt), self._to_tensor(y_gt, torch.long)
        x_hat_t, y_hat_t = self._to_tensor(x_hat), self._to_tensor(y_hat, torch.long)
        x_unl_t, y_unl_t = self._to_tensor(x_unl), self._to_tensor(y_unl, torch.long)
        w_t = (
            self._to_tensor(self.importance_weights)
            if self.importance_weights is not None
            else None
        )

        if self.optimizer == "lbfgs":
            model_params = optimize_ppi_lbfgs(
                self.model,
                x_gt=x_gt_t,
                y_gt=y_gt_t,
                x_hat=x_hat_t,
                y_hat=y_hat_t,
                x_unl=x_unl_t,
                y_unl=y_unl_t,
                w=w_t,
                params0=self.model_params,
                lambd_=lambd_,
                **self.optimizer_kwargs,
            )
        elif self.optimizer == "gd":
            model_params = optimize_ppi_gd(
                self.model,
                x_gt=x_gt_t,
                y_gt=y_gt_t,
                x_hat=x_hat_t,
                y_hat=y_hat_t,
                x_unl=x_unl_t,
                y_unl=y_unl_t,
                w=w_t,
                params0=self.model_params,
                lambd_=lambd_,
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        self.model_params = model_params
        return np.hstack(
            [model_params[name].cpu().numpy().reshape(-1) for name, _ in self.param_spec]
        )

    def grad_fn(
        self,
        inputs: tuple[np.ndarray, np.ndarray],
        w: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        x, y = inputs
        n_obs = x.shape[0]

        def likelihood(params, x_, y_, w_):
            return functional_call(self.model, params, (x_, y_, w_))["loss"]

        score = vmap(grad(likelihood), in_dims=(None, 0, 0, 0))
        all_grads = np.zeros((n_obs, self.n_params))
        for i in tqdm(range(0, n_obs, batch_size), desc="Gradient computation"):
            x_batch = self._to_tensor(x[i : i + batch_size])
            y_batch = self._to_tensor(y[i : i + batch_size], torch.long)
            n_obs_batch = x_batch.shape[0]
            w_batch = (
                self._to_tensor(w[i : i + batch_size])
                if w is not None
                else torch.ones(n_obs_batch, device=self.device)
            )
            grads = score(self.model_params, x_batch, y_batch, w_batch)
            stacked = [
                grads[name].reshape(n_obs_batch, -1).cpu().numpy() for name, _ in self.param_spec
            ]
            all_grads[i : i + batch_size] = np.hstack(stacked)
        return all_grads

    def hessian_fn(
        self,
        inputs: tuple[np.ndarray, np.ndarray],
        importance_weights: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        x, y = inputs
        n_obs = x.shape[0]

        def likelihood(params, x_, y_, w_):
            return functional_call(self.model, params, (x_, y_, w_))["loss"].sum()

        hess_fn = vmap(hessian(likelihood), in_dims=(None, 0, 0, 0))
        names = [name for name, _ in self.param_spec]
        sizes = [self._param_sizes[name] for name in names]

        hessian_sum = np.zeros((self.n_params, self.n_params), dtype=np.float64)
        for i in tqdm(range(0, n_obs, batch_size), desc="Hessian computation"):
            x_batch = self._to_tensor(x[i : i + batch_size]).unsqueeze(1)
            y_batch = self._to_tensor(y[i : i + batch_size], torch.long).unsqueeze(1)
            n_obs_batch = x_batch.shape[0]
            if importance_weights is not None:
                w_batch = self._to_tensor(importance_weights[i : i + batch_size]).unsqueeze(1)
            else:
                w_batch = torch.ones(n_obs_batch, 1, device=self.device)

            with warnings.catch_warnings():
                # Benign: torch.func.vmap(hessian(...)) triggers spurious internal PyTorch
                # warnings on this torch version (2.12.1): a UserWarning about autograd tensor
                # resize, and a DeprecationWarning about torch.jit.script. Both verified against
                # a naive per-observation loop (max abs diff ~1e-7, float32 noise) during planning.
                warnings.filterwarnings("ignore", message=_HESSIAN_VMAP_RESIZE_WARNING)
                warnings.filterwarnings(
                    "ignore",
                    message=_HESSIAN_VMAP_JIT_DEPRECATION_WARNING,
                    category=DeprecationWarning,
                )
                hess = hess_fn(self.model_params, x_batch, y_batch, w_batch)

            blocks = [
                [
                    hess[p1][p2].mean(0).reshape(sizes[i1], sizes[i2]).cpu().numpy()
                    for i2, p2 in enumerate(names)
                ]
                for i1, p1 in enumerate(names)
            ]
            full = np.block(blocks)
            hessian_sum += full * (n_obs_batch / n_obs)
        return hessian_sum

    def _construct_contrast(self, feature_id: int, idx_a: int) -> np.ndarray:
        parts = []
        for name, shape in self.param_spec:
            if name == "mu":
                block = np.zeros(shape)
                block[idx_a - 1, feature_id] = 1.0
                parts.append(block.flatten())
            else:
                parts.append(np.zeros(shape).flatten())
        return np.hstack(parts)

    def idx_to_feat(self) -> np.ndarray:
        parts = []
        for name, shape in self.param_spec:
            if name == "mu":
                block = np.ones(shape) * np.arange(self.n_features)
                parts.append(block.flatten())
            else:
                parts.append(np.arange(self.n_features))
        return np.hstack(parts).astype(int)

    def construct_contrast(self, idx_a: int) -> np.ndarray:
        return np.vstack([self._construct_contrast(fid, idx_a) for fid in range(self.n_features)])

    def _get_param_id(
        self, feature_id: int = None, class_id: int = None, param_type: str = None
    ) -> int:
        offset = 0
        for name, _shape in self.param_spec:
            size = self._param_sizes[name]
            if name == param_type:
                if name == "mu":
                    return offset + (class_id - 1) * self.n_features + feature_id
                return offset + feature_id
            offset += size
        raise ValueError(f"Unknown param_type: {param_type}")

    def _get_param_mask(self, feature_id: int) -> np.ndarray:
        ids = []
        for name, _ in self.param_spec:
            if name == "mu":
                ids.extend(
                    self._get_param_id(feature_id=feature_id, class_id=class_id, param_type="mu")
                    for class_id in range(1, self.n_classes)
                )
            else:
                ids.append(self._get_param_id(feature_id=feature_id, param_type=name))
        return np.hstack(ids)

    def get_beta(self, idx_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx_a == 0:
            raise ValueError("`class_a` cannot be the reference class.")
        contrast = self.construct_contrast(idx_a)
        beta = contrast @ self.theta
        cov = contrast @ self.sigma @ contrast.T
        return beta, cov, contrast

    def test_differential_expression(
        self,
        idx_a: int,
        feature_names: list[str] | None = None,
        cond_thresh: float = np.inf,
    ) -> pd.DataFrame:
        idx_a_ = idx_a - 1
        results = []
        for feature_id in range(self.n_features):
            mask_ = self._get_param_mask(feature_id)
            v_ = self.v[mask_][:, mask_]
            hess_ = self.hessian[mask_][:, mask_]

            beta = self.theta[mask_]
            cov = self._compute_sigma(hess_, v_, self.n)
            cond = np.linalg.cond(hess_)

            if cond >= cond_thresh:
                pval = 1.0
            else:
                _, pval = _zstat_generic2(
                    beta[idx_a_], np.sqrt(cov[idx_a_, idx_a_]), alternative="two-sided"
                )
            results.append(
                {
                    "pval": pval,
                    "hess": hess_[idx_a_, idx_a_],
                    "beta": beta[idx_a_],
                    "cov": cov[idx_a_, idx_a_],
                    "hess_cond": cond,
                }
            )
        res = pd.DataFrame(results)
        res.loc[np.isnan(res["pval"]), "pval"] = 1.0
        res["padj"] = multipletests(res["pval"], method="fdr_bh")[1]
        res["is_significant_005"] = res["padj"] < 0.05
        if feature_names is not None:
            res["feature_name"] = feature_names
        return res
