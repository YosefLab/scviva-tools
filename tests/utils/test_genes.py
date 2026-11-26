"""Tests for matching spatial genes to the genes of a reference model."""

from __future__ import annotations

import numpy as np
import pytest

from scviva.utils import match_reference_genes


def _match(st, ref):
    return match_reference_genes(st, ref, model_name="Model", allow_subset=True)


def test_match_reference_genes_identical_returns_none():
    assert _match(["a", "b"], ["a", "b"]) is None


def test_match_reference_genes_subset_positions():
    np.testing.assert_array_equal(_match(["c", "a"], ["a", "b", "c"]), [2, 0])


def test_match_reference_genes_duplicated_reference_not_requested():
    """Duplicates among reference genes that are not requested do not block the match."""
    np.testing.assert_array_equal(_match(["d", "a"], ["a", "b", "b", "d"]), [3, 0])


def test_match_reference_genes_duplicated_reference_requested_raises():
    with pytest.raises(ValueError, match="match several genes"):
        _match(["b", "a"], ["a", "b", "b"])


def test_match_reference_genes_absent_raises():
    with pytest.raises(ValueError, match="absent from the reference model"):
        _match(["a", "z"], ["a", "b", "b"])


def test_match_reference_genes_duplicated_spatial_raises():
    with pytest.raises(ValueError, match="not unique"):
        _match(["a", "a"], ["a", "b"])
