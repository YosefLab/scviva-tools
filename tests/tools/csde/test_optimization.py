import pytest
import torch

from scviva.tools.csde._optimization import _zstat_generic2, optimize_ppi_gd, optimize_ppi_lbfgs


def test_zstat_generic2_two_sided_matches_scipy():
    from scipy import stats

    zstat, pval = _zstat_generic2(value=2.0, std=1.0, alternative="two-sided")
    assert zstat == pytest.approx(2.0)
    assert pval == pytest.approx(stats.norm.sf(2.0) * 2)


def test_zstat_generic2_invalid_alternative_raises():
    with pytest.raises(ValueError):
        _zstat_generic2(value=1.0, std=1.0, alternative="bogus")


class _LinearGaussianModel(torch.nn.Module):
    """Toy model: y = w*x, Gaussian NLL loss, used only to exercise the optimizer plumbing."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, y, w=None):
        pred = self.w * x
        loss_unsummed = (pred - y) ** 2
        loss = loss_unsummed.sum(-1)
        if w is not None:
            loss = loss * w
            loss_unsummed = loss_unsummed * w[..., None]
        return {"loss": loss, "loss_unsummed": loss_unsummed}


@pytest.fixture
def toy_data():
    torch.manual_seed(0)
    true_w = 3.0
    x = torch.rand(50, 1) * 2
    y = true_w * x + 0.01 * torch.randn(50, 1)
    return x, y, true_w


def test_optimize_ppi_gd_recovers_slope(toy_data):
    x, y, true_w = toy_data
    model = _LinearGaussianModel()
    params = optimize_ppi_gd(
        model,
        x_gt=x,
        y_gt=y,
        x_hat=x,
        y_hat=y,
        x_unl=x,
        y_unl=y,
        lambd_=1.0,
        n_iter=500,
        learning_rate=0.1,
        optimizer="adam",
    )
    assert params["w"].item() == pytest.approx(true_w, abs=0.05)


def test_optimize_ppi_lbfgs_recovers_slope(toy_data):
    x, y, true_w = toy_data
    model = _LinearGaussianModel()
    params = optimize_ppi_lbfgs(
        model, x_gt=x, y_gt=y, x_hat=x, y_hat=y, x_unl=x, y_unl=y, lambd_=1.0
    )
    assert params["w"].item() == pytest.approx(true_w, abs=0.05)
