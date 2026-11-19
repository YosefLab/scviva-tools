from __future__ import annotations

import torch


def resolve_device(device: torch.device | str) -> torch.device:
    """Resolve device string, preferring CUDA, then MPS, then CPU in "auto" mode.

    Parameters
    ----------
    device
        Requested device. ``None`` or ``"auto"`` selects CUDA if available, else MPS if
        available, else CPU. Any other string is passed to :class:`torch.device`. If the
        resolved device is CUDA or MPS but a test tensor cannot be allocated on it, CPU is
        used instead.

    Returns
    -------
    The resolved :class:`torch.device`.
    """
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

    Parameters
    ----------
    device
        Device (or device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``) on which the
        statistics will be computed.

    Returns
    -------
    ``torch.float32`` if ``device`` is an MPS device, otherwise ``torch.float64``.
    """
    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    return torch.float32 if device_type == "mps" else torch.float64
