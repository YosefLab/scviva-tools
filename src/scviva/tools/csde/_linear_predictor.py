import torch


def per_class_linear_predictor(
    mu0: torch.Tensor, mu: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Per-class-intercept linear predictor: mu0 + one_hot(y) @ mu, with class 0 pinned to 0.

    `mu0` has shape (n_features,), `mu` has shape (n_classes-1, n_features), `y` has shape
    (n_obs,) with integer class labels in [0, n_classes). Returns shape (n_obs, n_features).

    Uses one-hot @ matmul (not fancy indexing `mu_full[y]`) because fancy indexing breaks
    under `torch.func.vmap(grad(...))` for batched integer-index gathers -- verified during
    the JAX->torch port. Shared by PoissonInterceptModule and NBInterceptModule.
    """
    n_classes = mu.shape[0] + 1
    mu_placeholder = torch.zeros_like(mu0)
    mu_full = torch.cat([mu_placeholder[None], mu], dim=0)  # (n_classes, n_features)
    class_ids = torch.arange(n_classes, device=y.device)
    y_oh = (class_ids == y.unsqueeze(-1)).to(mu_full.dtype)  # (n_obs, n_classes)
    return y_oh @ mu_full + mu0  # (n_obs, n_features)
