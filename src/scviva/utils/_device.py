from __future__ import annotations

import torch


def resolve_device(device: torch.device | str) -> torch.device:
    """Resolve device string, preferring CUDA, then MPS, then CPU in "auto" mode."""
    if device is None or (isinstance(device, str) and device == "auto"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if device.type in ("cuda", "mps"):
        try:
            torch.tensor([0.0], device=device)
        except (RuntimeError, AssertionError):
            device = torch.device("cpu")
    return device


def stats_dtype(device: torch.device | str) -> torch.dtype:
    """Preferred floating dtype for statistical computations on ``device``.

    MPS has no float64 kernel support (a Metal hardware limitation, not a
    version gap), so double-precision statistics fall back to float32 there;
    CPU and CUDA keep float64 for numerical stability.
    """
    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    return torch.float32 if device_type == "mps" else torch.float64
