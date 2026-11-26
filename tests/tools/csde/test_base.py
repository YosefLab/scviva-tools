import numpy as np
import pytest

from scviva.tools.csde._base import PPIAbstractClass


class _DummyPPI(PPIAbstractClass):
    """Minimal concrete subclass exercising the numpy-only sandwich-variance math."""

    def get_pointestimate(self, lambd_):
        return np.array([1.0, 2.0])

    def grad_fn(self, inputs):
        x, _y = inputs
        rng = np.random.default_rng(0)
        return rng.normal(size=(x.shape[0], 2))

    def hessian_fn(self, inputs):
        return np.eye(2)


@pytest.fixture
def dummy_ppi():
    rng = np.random.default_rng(0)
    x_gt, y_gt = rng.normal(size=(20, 3)), rng.integers(0, 2, size=20)
    x_unl, y_unl = rng.normal(size=(80, 3)), rng.integers(0, 2, size=80)
    return _DummyPPI(inputs_gt=(x_gt, y_gt), inputs_hat=(x_gt, y_gt), inputs_unl=(x_unl, y_unl))


def test_n_N_r(dummy_ppi):
    assert dummy_ppi.n == 20
    assert dummy_ppi.N == 80
    assert dummy_ppi.r == pytest.approx(0.25)


def test_get_asymptotic_distribution_returns_theta_and_sigma(dummy_ppi):
    dummy_ppi.lambd_ = 1.0
    theta, sigma = dummy_ppi.get_asymptotic_distribution()
    assert theta is None  # theta only set by fit(), which this dummy never calls
    assert sigma.shape == (2, 2)


def test_compute_sigma_is_symmetric_psd(dummy_ppi):
    sigma = dummy_ppi.compute_sigma(lambd=1.0)
    assert np.allclose(sigma, sigma.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(sigma)
    assert (eigvals >= -1e-8).all()


def test_abstract_methods_raise_on_base_class():
    base = PPIAbstractClass.__new__(PPIAbstractClass)
    with pytest.raises(NotImplementedError):
        PPIAbstractClass.get_pointestimate(base, lambd_=1.0)
    with pytest.raises(NotImplementedError):
        PPIAbstractClass.grad_fn(base, inputs=None)
    with pytest.raises(NotImplementedError):
        PPIAbstractClass.hessian_fn(base, inputs=None)
