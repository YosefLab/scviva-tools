from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "src")]

# -- Project information -----------------------------------------------------

info = metadata("scviva-tools")
project_name = "scviva-tools"
author = "Ori Kronfeld"
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True


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
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
myst_enable_extensions = ["colon_fence", "dollarmath", "html_image"]
myst_heading_anchors = 3
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers"]

# sphinx-autodoc-typehints inspects torch.distributed.reduce_op, which torch itself
# deprecated in favor of ReduceOp; this is upstream noise, not actionable here.
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="`torch.distributed.reduce_op` is deprecated"
)

# scvi-tools 1.5.0 has two invalid TYPE_CHECKING imports in PyroSampleMixin:
# PyroBaseModuleClass is imported from pyro instead of scvi.module.base, and
# AnnDataLoader is left unresolved as a result. These warnings are emitted while
# autodoc documents ResolVI's inherited sample_posterior method.


class _ScviPyroTypeHintFilter(logging.Filter):
    _messages = (
        "Failed guarded type import with ImportError("
        "\"cannot import name 'PyroBaseModuleClass' from 'pyro'",
        "Cannot resolve forward reference in type annotations of "
        "\"scviva.model.ResolVI.sample_posterior\": name 'AnnDataLoader' is not defined",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._messages)


_scvi_pyro_type_hint_filter = _ScviPyroTypeHintFilter()
logging.getLogger("sphinx.sphinx_autodoc_typehints").addFilter(_scvi_pyro_type_hint_filter)

# -- HTML output -------------------------------------------------------------

html_logo = "_static/logo.svg"

html_theme = "sphinx_book_theme"
html_title = "scVIVA-Tools"
html_context = {
    "display_github": True,
    "github_user": "YosefLab",
    "github_repo": "scviva-tools",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_theme_options = {
    "repository_url": "https://github.com/YosefLab/scviva-tools",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "navbar_persistent": [],
    "repository_branch": "main",
    "path_to_docs": "docs",
}

html_static_path = ["_static"]
html_css_files = ["css/override.css"]
html_js_files = ["js/custom.js"]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "scvi": ("https://docs.scvi-tools.org/en/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
