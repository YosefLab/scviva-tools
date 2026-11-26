import numpy as np
import pytest
import torch

from scviva.tools.csde._intercept_model import InterceptPPIModel


class _ToyPoissonModule(torch.nn.Module):
    """Minimal per-class-intercept Poisson module for exercising InterceptPPIModel in isolation.
    Structurally identical to Task 4's real PoissonInterceptModule.
    """

    def __init__(self, n_classes: int, n_features: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.mu0 = torch.nn.Parameter(torch.zeros(n_features))
        self.mu = torch.nn.Parameter(torch.zeros(n_classes - 1, n_features))

    def forward(self, x, y, w=None):
        mu_placeholder = torch.zeros_like(self.mu0)
        mu = torch.cat([mu_placeholder[None], self.mu], dim=0)
        class_ids = torch.arange(self.n_classes, device=x.device)
        y_oh = (class_ids == y.unsqueeze(-1)).to(mu.dtype)
        mus_ = y_oh @ mu + self.mu0
        if w is None:
            w = torch.ones_like(y, dtype=x.dtype)
        rates = torch.exp(mus_)
        log_px_c_unsummed = torch.distributions.Poisson(rate=rates, validate_args=False).log_prob(
            x
        )
        log_px_c = log_px_c_unsummed.sum(-1)
        loss = -log_px_c
        return {"loss": loss * w, "loss_unsummed": -log_px_c_unsummed * w[..., None]}


class _ToyPoissonIntercept(InterceptPPIModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = _ToyPoissonModule(n_classes=self.n_classes, n_features=self.n_features).to(
            self.device
        )
        self.param_spec = [
            ("mu", (self.n_classes - 1, self.n_features)),
            ("mu0", (self.n_features,)),
        ]
        self._finalize_init()


@pytest.fixture
def synthetic_data():
    np.random.seed(0)
    torch.manual_seed(0)
    n_features = 6
    n_gt, n_unl = 80, 300
    true_mu0 = np.random.uniform(0.5, 1.5, n_features)
    true_mu = np.array(
        [
            np.random.uniform(-0.5, 0.5, n_features),
            np.random.uniform(-0.3, 0.3, n_features),
        ]
    )

    def simulate(n, y):
        rates = np.exp(true_mu0 + (y[:, None] == 1) * true_mu[0] + (y[:, None] == 2) * true_mu[1])
        return np.random.poisson(rates).astype(float)

    y_gt = np.random.choice([0, 1, 2], size=n_gt)
    x_gt = simulate(n_gt, y_gt)
    y_unl = np.random.choice([0, 1, 2], size=n_unl)
    x_unl = simulate(n_unl, y_unl)
    return x_gt, y_gt, x_unl, y_unl, true_mu, n_features


def test_n_params_matches_param_spec(synthetic_data):
    x_gt, y_gt, x_unl, y_unl, _true_mu, n_features = synthetic_data
    model = _ToyPoissonIntercept(
        inputs_gt=(x_gt, y_gt), inputs_hat=(x_gt, y_gt), inputs_unl=(x_unl, y_unl)
    )
    n_classes = len(np.unique(y_gt))
    assert model.n_params == (n_classes - 1) * n_features + n_features


def test_fit_recovers_true_effect(synthetic_data):
    x_gt, y_gt, x_unl, y_unl, true_mu, n_features = synthetic_data
    model = _ToyPoissonIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),
        inputs_unl=(x_unl, y_unl),
        optimizer="gd",
        optimizer_kwargs={"n_iter": 300, "learning_rate": 0.05},
    )
    model.fit(lambd_=None)
    model.get_asymptotic_distribution()
    recovered_mu1 = model.theta[:n_features]
    # exact values verified during planning: max abs diff ~0.1547 for this seed/config
    assert np.max(np.abs(recovered_mu1 - true_mu[0])) < 0.2


def test_test_differential_expression_shape_and_no_nans(synthetic_data):
    x_gt, y_gt, x_unl, y_unl, _true_mu, n_features = synthetic_data
    model = _ToyPoissonIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),
        inputs_unl=(x_unl, y_unl),
        optimizer="gd",
        optimizer_kwargs={"n_iter": 300, "learning_rate": 0.05},
    )
    model.fit(lambd_=None)
    model.get_asymptotic_distribution()
    res = model.test_differential_expression(
        idx_a=1, feature_names=[f"g{i}" for i in range(n_features)]
    )
    assert len(res) == n_features
    assert not res["pval"].isnull().any()
    assert "feature_name" in res.columns


def test_bad_importance_weights_shape_raises():
    x_gt = np.random.poisson(2.0, size=(10, 3)).astype(float)
    y_gt = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        _ToyPoissonIntercept(
            inputs_gt=(x_gt, y_gt),
            inputs_hat=(x_gt, y_gt),
            inputs_unl=(x_gt, y_gt),
            importance_weights=np.ones(15),
        )
