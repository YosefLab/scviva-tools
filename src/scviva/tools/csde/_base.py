import numpy as np


class PPIAbstractClass:
    """Prediction-powered inference (PPI) core: sandwich-variance estimation.

    Shared by all CSDE noise models. Pure numpy — ported unchanged from upstream
    CSDE's `_base.py`, which has no JAX dependency.
    """

    def __init__(
        self,
        inputs_gt: tuple[np.ndarray, np.ndarray] | np.ndarray,
        inputs_hat: tuple[np.ndarray, np.ndarray] | np.ndarray,
        inputs_unl: tuple[np.ndarray, np.ndarray] | np.ndarray,
        lambd_mode: str = "overall",
    ):
        self.inputs_gt = inputs_gt
        self.inputs_hat = inputs_hat
        self.inputs_unl = inputs_unl

        inputs_are_tuples = isinstance(inputs_gt, tuple)
        if inputs_are_tuples:
            self.n = inputs_gt[0].shape[0]
            self.N = inputs_unl[0].shape[0]
        else:
            self.n = self.inputs_gt.shape[0]
            self.N = self.inputs_unl.shape[0]
        self.r = float(self.n) / self.N
        self.theta = None
        self.sigma = None
        self.hessian = None
        self.v = None
        self.lambd_mode = lambd_mode
        self.lambd_ = None

    def get_asymptotic_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        self.sigma = self.compute_sigma(self.lambd_)
        return self.theta, self.sigma

    def compute_sigma(self, lambd: float | np.ndarray) -> np.ndarray:
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (lambd**2) * (grad_f_.T @ grad_f_) / (self.n + self.N)
        rect_ = grad_f_gt - lambd * grad_f_hat
        rect_ = rect_ - rect_.mean(axis=0)
        vdelta = (rect_.T @ rect_) / self.n
        v = vdelta + (self.r * vf)

        hess = self.hessian_fn(self.inputs_gt)
        self.hessian = hess
        self.v = v
        return self._compute_sigma(hess, v, self.n)

    @staticmethod
    def _compute_sigma(hess: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
        inv_hess = np.linalg.pinv(hess)
        sigma_ = inv_hess @ v @ inv_hess
        sigma_ = sigma_ / n
        return sigma_

    def get_lambda(
        self,
        lambd_0: float = 0.5,
        idx_to_optimize: int | list[int] | None = None,
    ) -> float | np.ndarray:
        self.theta = self.get_pointestimate(lambd_=lambd_0)

        hess = self.hessian_fn(self.inputs_gt)
        inv_hess = np.linalg.pinv(hess)
        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)

        grad_f_hat_ = grad_f_hat - grad_f_hat.mean(0)
        grad_f_gt_ = grad_f_gt - grad_f_gt.mean(0)
        cov1 = (grad_f_hat_.T @ grad_f_gt_) / self.n
        cov2 = (grad_f_gt_.T @ grad_f_hat_) / self.n

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        num = inv_hess @ (cov1 + cov2) @ inv_hess
        denom = 2 * (1.0 + self.r) * (inv_hess @ vf @ inv_hess)
        if idx_to_optimize is not None:
            if isinstance(idx_to_optimize, int):
                return (
                    num[idx_to_optimize, idx_to_optimize] / denom[idx_to_optimize, idx_to_optimize]
                )
            return np.trace(num[idx_to_optimize, :][:, idx_to_optimize]) / np.trace(
                denom[idx_to_optimize, :][:, idx_to_optimize]
            )
        return np.trace(num) / np.trace(denom)

    def get_pointestimate(self, lambd_: float) -> np.ndarray:
        raise NotImplementedError

    def grad_fn(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def hessian_fn(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        raise NotImplementedError
