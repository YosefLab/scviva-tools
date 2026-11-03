from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE / "extensions")]

# -- Project information -----------------------------------------------------

info = metadata("scviva-tools")
project = "scviva-tools"
author = "Ori Kronfeld"
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
release = version
source_repository = "https://github.com/YosefLab/scviva-tools"

# -- General configuration ---------------------------------------------------

# scverse_doc sets up myst_nb, autodoc, autosummary, intersphinx, napoleon, sphinx_copybutton,
# sphinx_design and sphinxext.opengraph, and selects the scverse theme.
extensions = [
    "scverse_doc",
    "sphinx.ext.linkcode",  # [source] links to GitHub, resolved by scverse_doc
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]


def _torch_inherited_members() -> list[str]:
    """Return ``"<class fullname>.<member>"`` for members scviva.module classes get from torch.

    Used by _templates/autosummary/class.rst to hide the dozens of members that
    nn.Module subclasses inherit from torch (e.g. ``cuda``, ``zero_grad``). Kept as a
    plain list so Sphinx can pickle the config.
    """
    import scviva.module

    out = []
    for name in scviva.module.__all__:
        cls = getattr(scviva.module, name)
        for member in dir(cls):
            owner = next((c for c in cls.__mro__ if member in vars(c)), None)
            if owner is not None and owner.__module__.startswith("torch"):
                out.append(f"scviva.module.{name}.{member}")
    return sorted(out)


autosummary_context = {"torch_inherited": _torch_inherited_members()}
autodoc_member_order = "bysource"
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
napoleon_google_docstring = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
myst_heading_anchors = 3

templates_path = ["_templates"]
# scverse-doc turns nitpicky mode on; the docstrings are not yet clean enough for it.
nitpicky = False
exclude_patterns = [*DEFAULTS.exclude_patterns, "superpowers"]

# sphinx-autodoc-typehints inspects torch.distributed.reduce_op, which torch itself
# deprecated in favor of ReduceOp; this is upstream noise, not actionable here.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="`torch.distributed.reduce_op` is deprecated"
)

# scvi-tools 1.5.0 has two invalid TYPE_CHECKING imports in PyroSampleMixin:
# PyroBaseModuleClass is imported from pyro instead of scvi.module.base, and
# AnnDataLoader is left unresolved as a result. These warnings are emitted while
# autodoc documents ResolVI's inherited sample_posterior method.
# VAEMixin.get_aggregated_posterior is also annotated ``-> torch.dist.Distribution``
# (``torch.dist`` is a function), which newer sphinx-autodoc-typehints cannot resolve
# for the models that inherit it (SCVIVA, DIAGVI). Fixed upstream in scverse/scvi-tools#4091;
# drop that entry once a scvi-tools release with the fix is required.


class _ScviPyroTypeHintFilter(logging.Filter):
    _messages = (
        "Failed guarded type import with ImportError("
        "\"cannot import name 'PyroBaseModuleClass' from 'pyro'",
        "Cannot resolve forward reference in type annotations of "
        "\"scviva.model.ResolVI.sample_posterior\": name 'AnnDataLoader' is not defined",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith(self._messages):
            return False
        return not (
            '.get_aggregated_posterior" (module scvi.model.base._vaemixin)' in message
            and "'builtin_function_or_method' object has no attribute 'Distribution'" in message
        )


_scvi_pyro_type_hint_filter = _ScviPyroTypeHintFilter()
# Logger filters do not apply to records of child loggers, and newer sphinx-autodoc-typehints
# emits its categorized warnings from the `_warnings` submodule's logger.
for _logger_name in (
    "sphinx.sphinx_autodoc_typehints",
    "sphinx.sphinx_autodoc_typehints._warnings",
):
    logging.getLogger(_logger_name).addFilter(_scvi_pyro_type_hint_filter)

# -- HTML output -------------------------------------------------------------

html_logo = "_static/logo.svg"
html_title = "scVIVA-Tools"
# sphinxext-opengraph (set up by scverse_doc) cannot use the SVG logo for social cards.
ogp_social_cards = {"image": "_static/logo.png"}
html_theme_options = {
    # scviva-tools is not in the scverse registry, so use scvi-tools' yellow accent.
    "accent": "oklch(0.824 0.1634 81)",
    # The scverse theme's header, plus the Colab/download/fullscreen buttons that
    # sphinx-book-theme used to provide (see extensions/header_buttons.py).
    "article_header_end": [
        "header-buttons",
        "theme-switcher",
        "navbar-icon-links",
        "scverse-ecosystem",
        "toggle-secondary-sidebar",
    ],
}
# Copy sources under their own suffix, so the download button serves e.g. a usable .ipynb.
html_sourcelink_suffix = ""
# Keep MathJax rather than scverse-doc's math-core, which lacks \tag and \ref support.
html_math_renderer = "mathjax"

html_static_path = ["_static"]
html_css_files = ["css/override.css"]
html_js_files = ["js/custom.js"]

# -- Intersphinx mapping -----------------------------------------------------

# intersphinx() is scverse-doc's registry: it already maps python, numpy, scipy, pandas,
# matplotlib, anndata, scanpy, scvi-tools and the other scverse core packages, so only the rest
# is listed here.
intersphinx_mapping = intersphinx() | {
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
