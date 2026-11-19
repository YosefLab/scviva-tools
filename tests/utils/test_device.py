import torch

from scviva.utils import resolve_device, stats_dtype


def test_resolve_device_auto_falls_back_when_cuda_unusable(monkeypatch):
    """`cuda.is_available()` can be True while torch has no working CUDA build;
    `resolve_device` must fall back gracefully rather than raise. The allocation
    probe is forced to fail here rather than relying on the host genuinely lacking
    CUDA: on the real test_cuda.yaml GPU runner the probe would otherwise succeed
    (CUDA truly is usable there), making `resolve_device` correctly return "cuda"
    and this assertion fail."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def broken_probe(*args, **kwargs):
        raise RuntimeError("simulated: CUDA reports available but is not usable")

    monkeypatch.setattr(torch, "tensor", broken_probe)
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_auto_prefers_mps_over_cpu(monkeypatch):
    """Only the availability *preference* is under test here, not real hardware:
    `resolve_device` also probes the device with a real allocation
    (`torch.tensor(..., device=...)`) before committing to it, which would fail on
    any host without actual MPS (e.g. the ubuntu-latest CI matrix) and silently fall
    back to cpu, defeating the assertion below. So the probe is mocked out too; its
    return value is discarded by `resolve_device`, so a no-op stand-in is enough."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: None)
    assert resolve_device("auto").type == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_explicit_string():
    assert resolve_device("cpu") == torch.device("cpu")


def test_stats_dtype_mps_is_float32():
    assert stats_dtype(torch.device("mps")) == torch.float32


def test_stats_dtype_cpu_is_float64():
    assert stats_dtype(torch.device("cpu")) == torch.float64


def test_stats_dtype_accepts_device_string():
    assert stats_dtype("mps") == torch.float32
    assert stats_dtype("cpu") == torch.float64
