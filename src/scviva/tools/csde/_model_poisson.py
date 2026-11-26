import torch

from ._intercept_model import InterceptPPIModel
from ._linear_predictor import per_class_linear_predictor


class PoissonInterceptModule(torch.nn.Module):
    """Per-class-intercept Poisson GLM: log-rate = mu0 + one_hot(y) @ mu (class 0 pinned to 0).

    Torch port of upstream's `flax.linen`-based `PoissonInterceptModule`. Uses one-hot @ matmul
    (not fancy indexing `mu[y]`) because `mu_full[y]` breaks under `torch.func.vmap(grad(...))`
    for batched integer-index gathers — verified during the JAX->torch port (see plan/spec).
    """

    def __init__(self, n_classes: int, n_features: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.mu0 = torch.nn.Parameter(torch.zeros(n_features))
        self.mu = torch.nn.Parameter(torch.zeros(n_classes - 1, n_features))

    def forward(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None):
        mus_ = per_class_linear_predictor(self.mu0, self.mu, y)

        if w is None:
            w = torch.ones_like(y, dtype=x.dtype)

        rates = torch.exp(mus_)
        log_px_c_unsummed = torch.distributions.Poisson(rate=rates, validate_args=False).log_prob(
            x
        )
        log_px_c = log_px_c_unsummed.sum(-1)

        loss = -log_px_c
        return {
            "loss": loss * w,
            "loss_unsummed": -log_px_c_unsummed * w[..., None],
        }


class PoissonIntercept(InterceptPPIModel):
    """Poisson noise model for CSDE's prediction-powered DE test.

    All PPI machinery (fit, grad_fn, hessian_fn, test_differential_expression, ...) is inherited
    from `InterceptPPIModel`; this class only wires up the Poisson-specific module and param
    layout.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = PoissonInterceptModule(
            n_classes=self.n_classes, n_features=self.n_features
        ).to(self.device)
        self.param_spec = [
            ("mu", (self.n_classes - 1, self.n_features)),
            ("mu0", (self.n_features,)),
        ]
        self._finalize_init()
