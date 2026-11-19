import numpy as np
import pytest
import torch

from scviva.tools.csde._model_poisson import PoissonIntercept


@pytest.fixture
def synthetic_poisson_data():
    np.random.seed(0)
    torch.manual_seed(0)
    n_features = 6
    n_gt, n_unl = 80, 300

    true_mu0 = np.random.uniform(0.5, 1.5, n_features)
    true_mu = np.array(
        [
            np.random.uniform(-0.5, 0.5, n_features),  # class 1 vs 0
            np.random.uniform(-0.3, 0.3, n_features),  # class 2 vs 0
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


def test_poisson_intercept_recovers_true_effect(synthetic_poisson_data):
    x_gt, y_gt, x_unl, y_unl, true_mu, n_features = synthetic_poisson_data

    model = PoissonIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),  # perfect predictions == gt for this recovery check
        inputs_unl=(x_unl, y_unl),
        optimizer="gd",
        optimizer_kwargs={"n_iter": 300, "learning_rate": 0.05},
    )
    model.fit(lambd_=None)
    model.get_asymptotic_distribution()

    recovered_mu1 = model.theta[:n_features]
    assert np.max(np.abs(recovered_mu1 - true_mu[0])) < 0.2


def test_poisson_intercept_pvalues_separate_null_from_effect(synthetic_poisson_data):
    x_gt, y_gt, x_unl, y_unl, true_mu, n_features = synthetic_poisson_data

    model = PoissonIntercept(
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
    assert list(res.columns[:5]) == ["pval", "hess", "beta", "cov", "hess_cond"]
    assert not res["pval"].isnull().any()
    # gene with the largest true effect should be far more significant than
    # whichever gene has the smallest true |effect|
    largest_effect_gene = int(np.argmax(np.abs(true_mu[0])))
    smallest_effect_gene = int(np.argmin(np.abs(true_mu[0])))
    assert res["pval"].iloc[largest_effect_gene] < res["pval"].iloc[smallest_effect_gene]


def test_poisson_intercept_lbfgs_uses_importance_weights(synthetic_poisson_data):
    x_gt, y_gt, x_unl, y_unl, true_mu, n_features = synthetic_poisson_data

    model_unweighted = PoissonIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),
        inputs_unl=(x_unl, y_unl),
        optimizer="lbfgs",
        importance_weights=None,
    )
    model_unweighted.fit(lambd_=1.0)
    assert np.isfinite(model_unweighted.theta).all()

    # strongly skewed, non-uniform weights so an unweighted vs. weighted fit must differ
    rng = np.random.default_rng(0)
    skewed_weights = np.where(y_gt == 0, 10.0, 0.1)
    skewed_weights = skewed_weights * rng.uniform(0.9, 1.1, size=skewed_weights.shape)

    model_weighted = PoissonIntercept(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_gt, y_gt),
        inputs_unl=(x_unl, y_unl),
        optimizer="lbfgs",
        importance_weights=skewed_weights,
    )
    model_weighted.fit(lambd_=1.0)
    assert np.isfinite(model_weighted.theta).all()

    # before the fix, lbfgs silently ignored importance_weights, so these two would be identical
    assert not np.allclose(model_unweighted.theta, model_weighted.theta)


def test_poisson_intercept_bad_importance_weights_shape_raises():
    x_gt = np.random.poisson(2.0, size=(10, 3)).astype(float)
    y_gt = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        PoissonIntercept(
            inputs_gt=(x_gt, y_gt),
            inputs_hat=(x_gt, y_gt),
            inputs_unl=(x_gt, y_gt),
            importance_weights=np.ones(15),
        )
