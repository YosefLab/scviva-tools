"""Article header buttons: launch in Colab, download the page, and full screen.

sphinx-book-theme provided these; the scverse theme (pydata-sphinx-theme) does not,
so this extension computes what ``docs/_templates/header-buttons.html`` needs.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx

#: Where the tutorial notebooks live, relative to the docs source directory.
NOTEBOOKS_DIR = PurePosixPath("tutorials")
#: Where the docs source directory lives, relative to the repository root.
DOCS_DIR = PurePosixPath("docs")
REPOSITORY = "YosefLab/scviva-tools"
COLAB_URL = "https://colab.research.google.com/github"


def _repository_ref(app: Sphinx) -> str:
    """The scviva-tools ref Colab opens.

    Releases are tagged ``v<version>``, so Read the Docs tag builds (e.g. ``stable``)
    open the matching tag and every other build opens ``main``.
    """
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag":
        return f"v{app.config.version}"
    return "main"


def add_header_button_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict,
    doctree: object,
) -> None:
    """Expose the Colab and download URLs to the header buttons template."""
    suffix = context.get("page_source_suffix")
    context["scvi_page_source_suffix"] = suffix
    context["scvi_colab_url"] = None
    if not suffix or doctree is None:
        return

    docpath = PurePosixPath(app.env.doc2path(pagename, base=False).as_posix())
    if suffix == ".ipynb" and NOTEBOOKS_DIR in docpath.parents:
        context["scvi_colab_url"] = (
            f"{COLAB_URL}/{REPOSITORY}/blob/{_repository_ref(app)}/{DOCS_DIR / docpath}"
        )


def setup(app: Sphinx) -> dict:
    """Setup the extension."""
    app.connect("html-page-context", add_header_button_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
