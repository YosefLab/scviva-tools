import numpy as np
import pytest
import torch
from scipy import stats

from scviva.tools.csde._model_nb import NBIntercept, _nb_log_prob


def test_nb_log_prob_matches_scipy_nbinom_parametrization():
    """Cross-check the NumPyro NegativeBinomial2(mean, concentration) -> torch reparam:
    probs = mean / (mean + concentration), total_count = concentration. Verified numerically
    during the JAX->torch port (the naive guess, probs = concentration/(mean+concentration),
    is backwards and silently produces the wrong likelihood).
    """
    mean = torch.tensor([3.5, 10.0, 0.5])
    concentration = torch.tensor([2.0, 5.0, 1.0])
    x = torch.tensor([2.0, 8.0, 0.0])

    lp_torch = _nb_log_prob(x, mean, concentration).numpy()

    probs_scipy_success = (mean / (mean + concentration)).numpy()
    lp_scipy = stats.nbinom.logpmf(x.numpy(), n=concentration.numpy(), p=1 - probs_scipy_success)
    assert np.allclose(lp_torch, lp_scipy, atol=1e-5)


@pytest.fixture
def synthetic_nb_data():
    np.random.seed(1)
    torch.manual_seed(1)
    n_features = 5
    n_gt, n_unl = 100, 300
    true_mu0 = np.random.uniform(0.5, 1.5, n_features)
    true_mu = np.random.uniform(-0.6, 0.6, n_features)
    true_conc = 5.0

    def simulate(n, y):
        rates = np.exp(true_mu0 + (y[:, None] == 1) * true_mu)
        lam = np.random.gamma(shape=true_conc, scale=rates / true_conc)
        return np.random.poisson(lam).astype(float)

    y_gt = np.random.choice([0, 1], size=n_gt)
    x_gt = simulate(n_gt, y_gt)
    y_unl = np.random.choice([0, 1], size=n_unl)
    x_unl = simulate(n_unl, y_unl)
    return x_gt, y_gt, x_unl, y_unl, true_mu, n_features


def test_nb_intercept_recovers_true_effect(synthetic_nb_data):
    x_gt, y_gt, x_unl, y_unl, true_mu, n_features = synthetic_nb_data
    model = NBIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),
        inputs_unl=(x_unl, y_unl),
        optimizer="gd",
        optimizer_kwargs={"n_iter": 400, "learning_rate": 0.03},
    )
    model.fit(lambd_=None)
    model.get_asymptotic_distribution()
    recovered_mu = model.theta[:n_features]
    assert np.max(np.abs(recovered_mu - true_mu)) < 0.2


def test_nb_intercept_lbfgs_runs_without_error(synthetic_nb_data):
    x_gt, y_gt, x_unl, y_unl, _true_mu, _n_features = synthetic_nb_data
    model = NBIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),
        inputs_unl=(x_unl, y_unl),
        optimizer="lbfgs",
    )
    model.fit(lambd_=0.5)
    assert model.theta is not None
    assert np.isfinite(model.theta).all()
