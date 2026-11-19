from typing import Any

import numpy as np
import scipy.stats as stats
import torch
from torch.func import functional_call


def _zstat_generic2(value: float, std: float, alternative: str) -> tuple[float, float]:
    """Two-sided/one-sided z-test p-value. Ported unchanged from upstream (pure scipy)."""
    zstat = value / std
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ["larger", "l"]:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ["smaller", "s"]:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError("invalid alternative")
    return zstat, pvalue


def optimize_ppi_gd(
    model: torch.nn.Module,
    x_gt: torch.Tensor,
    y_gt: torch.Tensor,
    x_hat: torch.Tensor,
    y_hat: torch.Tensor,
    x_unl: torch.Tensor,
    y_unl: torch.Tensor,
    w: torch.Tensor | None = None,
    params0: dict[str, torch.Tensor] | None = None,
    lambd_: float = 1.0,
    tol: float | None = 1e-3,
    n_iter: int = 10000,
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Optimize the PPI objective with Adam/SGD.

    Torch port of upstream's `optax`-based `optimize_ppi_gd`: minimizes `lambd_ *
    loss_unl - lambd_ * loss_hat + loss_gt`.
    """
    # NOTE: tol=None -> atol=inf, which makes np.isclose(...) trivially True on the very
    # first iteration, so the loop stops after 1 step rather than disabling early stopping.
    # This is inherited verbatim from upstream JAX CSDE's own optimize_ppi_gd (same quirk
    # there) -- kept deliberately for a faithful 1:1 port. No call site in this codebase
    # passes tol=None (default is 1e-3 everywhere), so this is currently inert.
    tol_ = np.inf if tol is None else tol

    if params0 is not None:
        params = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
    else:
        params = {k: v.clone().requires_grad_(True) for k, v in model.state_dict().items()}

    if optimizer == "adam":
        opt = torch.optim.Adam(list(params.values()), lr=learning_rate, **kwargs)
    elif optimizer == "gd":
        opt = torch.optim.SGD(list(params.values()), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    def loss_fn(p):
        loss_gt = functional_call(model, p, (x_gt, y_gt, w))["loss_unsummed"].mean(0)
        loss_hat = functional_call(model, p, (x_hat, y_hat, w))["loss_unsummed"].mean(0)
        loss_unl = functional_call(model, p, (x_unl, y_unl))["loss_unsummed"].mean(0)
        loss = (lambd_ * loss_unl) - (lambd_ * loss_hat) + loss_gt
        return loss.sum(-1)

    previous_loss = 1e6
    for i in range(n_iter):
        opt.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        opt.step()
        loss_val = loss.item()
        if np.isclose(loss_val, previous_loss, atol=tol_, rtol=0):
            if verbose:
                loss_diff = abs(loss_val - previous_loss)
                print(f"stopping criterion met at iter {i}: |loss diff|={loss_diff:.2e}")
            break
        previous_loss = loss_val
    return {k: v.detach() for k, v in params.items()}


def optimize_ppi_lbfgs(
    model: torch.nn.Module,
    x_gt: torch.Tensor,
    y_gt: torch.Tensor,
    x_hat: torch.Tensor,
    y_hat: torch.Tensor,
    x_unl: torch.Tensor,
    y_unl: torch.Tensor,
    w: torch.Tensor | None = None,
    params0: dict[str, torch.Tensor] | None = None,
    lambd_: float = 1.0,
    max_iter: int = 100,
    **lbfgs_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Optimize the PPI objective with L-BFGS.

    Torch port of upstream's `optax.lbfgs`-based `optimize_ppi`.
    """
    if params0 is not None:
        params = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
    else:
        params = {k: v.clone().requires_grad_(True) for k, v in model.state_dict().items()}

    opt = torch.optim.LBFGS(
        list(params.values()), line_search_fn="strong_wolfe", max_iter=max_iter, **lbfgs_kwargs
    )

    def loss_fn():
        loss_gt = functional_call(model, params, (x_gt, y_gt, w))["loss"].mean()
        loss_hat = functional_call(model, params, (x_hat, y_hat, w))["loss"].mean()
        loss_unl = functional_call(model, params, (x_unl, y_unl))["loss"].mean()
        return (lambd_ * loss_unl) - (lambd_ * loss_hat) + loss_gt

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    return {k: v.detach() for k, v in params.items()}
