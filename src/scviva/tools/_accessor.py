from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def adata_delegate(module: str, name: str, *, note: str | None = None) -> Callable[..., Any]:
    """Build an accessor method that forwards to ``module.name`` on the parent's AnnData.

    The returned method exposes the target's signature and docstring (with ``adata``
    made optional and the remaining parameters keyword-only), so ``help()``, IDEs and
    Sphinx show the real arguments instead of ``**kwargs``. The target is looked up at
    call time, so patching ``module.name`` is honoured.

    Parameters
    ----------
    module
        Dotted path of the module defining the target function.
    name
        Name of the target function. Its first parameter must be ``adata``.
    note
        Optional extra paragraph inserted after the forwarding note (e.g. a deprecation).
    """
    target = getattr(importlib.import_module(module), name)
    params = list(inspect.signature(target).parameters.values())
    if not params or params[0].name != "adata":
        raise TypeError(f"{module}.{name} must take `adata` as its first parameter")

    def method(self, adata=None, **kwargs):
        func = getattr(importlib.import_module(module), name)
        return func(self._resolve(adata), **kwargs)

    self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    adata_param = params[0].replace(default=None, annotation="AnnData | None")
    rest = [
        p if p.kind in (p.KEYWORD_ONLY, p.VAR_KEYWORD) else p.replace(kind=p.KEYWORD_ONLY)
        for p in params[1:]
        if p.kind != p.VAR_POSITIONAL
    ]
    method.__signature__ = inspect.signature(target).replace(
        parameters=[self_param, adata_param, *rest]
    )
    method.__name__ = name

    summary, _, body = inspect.cleandoc(target.__doc__ or "").partition("\n")
    forward = (
        f"Forwards to :func:`~{module}.{name}`. ``adata`` defaults to the AnnData "
        "held by the analysis object."
    )
    parts = [summary.strip(), forward]
    if note:
        parts.append(note)
    if body.strip():
        parts.append(body.strip("\n"))
    method.__doc__ = "\n\n".join(p for p in parts if p)
    return method
