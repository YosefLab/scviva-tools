from __future__ import annotations

import sys
from importlib import import_module
from importlib.metadata import version as _version

from scviva._settings import settings

# Eagerly import subpackages (as scvi-tools does) so `import scviva` alone makes
# `scviva.model.ResolVI`, `scviva.external.Tangram`, etc. resolvable as attributes.
# This import needs to come after `settings` to prevent circular imports.
from . import data, external, model, module, train, utils
from . import plotting as pl
from . import tools as tl

# Register `tl`/`pl` as real dotted-importable submodule aliases (mirrors scanpy's
# `sys.modules.update({f"{__name__}.{m}": globals()[m] for m in [...]})` trick), so
# single-hop access like `from scviva.tl import HarremanAnalysis` or
# `from scviva.tl import harreman` works, not just attribute access after `import
# scviva`. Do NOT import *through* `tl`/`pl` past that first hop (e.g.
# `scviva.tl.harreman._results`, `import scviva.tl.harreman.hotspot`) -- unlike
# `scviva.tools`/`scviva.plotting`, `tl`/`pl` aren't real packages with their own
# `__path__`-based submodule search, so resolving a name nested under them can
# make Python load a second, distinct copy of that submodule tree instead of
# reusing the one already loaded under `scviva.tools`/`scviva.plotting`. Anything
# deeper than one hop should use the real `scviva.tools.*`/`scviva.plotting.*` path.
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pl"]})

__version__ = _version("scviva-tools")

_MODEL_NAMES = {"SCVIVA", "DestVI", "ResolVI", "GIMVI"}


def __getattr__(name: str):
    if name in _MODEL_NAMES:
        mod = import_module("scviva.model")
        return getattr(mod, name)
    raise AttributeError(f"module 'scviva' has no attribute {name!r}")


__all__ = [
    "SCVIVA",
    "DestVI",
    "ResolVI",
    "GIMVI",
    "__version__",
    "data",
    "external",
    "model",
    "module",
    "pl",
    "settings",
    "tl",
    "train",
    "utils",
]
