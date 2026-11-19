import torch

from ._intercept_model import InterceptPPIModel
from ._linear_predictor import per_class_linear_predictor


def _nb_log_prob(x: torch.Tensor, mean: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
    """NumPyro's NegativeBinomial2(mean, concentration) reparameterized for torch.distributions.

    torch.distributions.NegativeBinomial(total_count=r, probs=p) counts the number of successes
    before r failures, with per-trial success probability p, giving mean = r*p/(1-p). Solving for
    p with r=concentration gives probs = mean / (mean + concentration), reproducing the NB2
    variance = mean + mean^2/concentration. Verified against scipy.stats.nbinom during the
    JAX->torch port — the mirror-image guess (probs = concentration/(mean+concentration)) is
    backwards and silently yields the wrong likelihood.
    """
    probs = mean / (mean + concentration)
    return torch.distributions.NegativeBinomial(
        total_count=concentration, probs=probs, validate_args=False
    ).log_prob(x)


class NBInterceptModule(torch.nn.Module):
    def __init__(self, n_classes: int, n_features: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.mu0 = torch.nn.Parameter(torch.zeros(n_features))
        self.mu = torch.nn.Parameter(torch.zeros(n_classes - 1, n_features))
        self.rho = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None):
        mus_ = per_class_linear_predictor(self.mu0, self.mu, y)
        rates = torch.exp(mus_)
        concentrations = torch.exp(self.rho)

        if w is None:
            w = torch.ones_like(y, dtype=x.dtype)

        log_px_c_unsummed = _nb_log_prob(x, rates, concentrations)
        log_px_c = log_px_c_unsummed.sum(-1)

        loss = -log_px_c
        return {
            "loss": loss * w,
            "loss_unsummed": -log_px_c_unsummed * w[..., None],
        }


class NBIntercept(InterceptPPIModel):
    """Negative-binomial noise model for CSDE's prediction-powered DE test.

    All PPI machinery is inherited from `InterceptPPIModel`; this class only wires up the
    NB-specific module and param layout (`rho` = per-feature log-concentration, in addition to
    Poisson's `mu`/`mu0`).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = NBInterceptModule(n_classes=self.n_classes, n_features=self.n_features).to(
            self.device
        )
        self.param_spec = [
            ("mu", (self.n_classes - 1, self.n_features)),
            ("mu0", (self.n_features,)),
            ("rho", (self.n_features,)),
        ]
        self._finalize_init()
