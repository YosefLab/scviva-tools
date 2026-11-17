"""Subpackages must be reachable as attributes after a plain ``import scviva``.

Run in a fresh interpreter so that submodules already imported by other tests
(or by pytest collection) can't mask a missing eager import.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ATTRIBUTE_PATHS = [
    "scviva.model.SCVIVA",
    "scviva.model.ResolVI",
    "scviva.model.DestVI",
    "scviva.model.GIMVI",
    "scviva.data.SpatialCoordsField",
    "scviva.external.Tangram",
    "scviva.module.JVAE",
    "scviva.train.GIMVITrainingPlan",
    "scviva.utils.resolve_device",
    "scviva.ResolVI",
    "scviva.tl",
    "scviva.pl",
]


@pytest.mark.parametrize("path", ATTRIBUTE_PATHS)
def test_attribute_access_after_plain_import(path: str):
    code = f"import scviva; {path}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
