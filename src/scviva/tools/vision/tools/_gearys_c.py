"""Geary's C autocorrelation, vendored from scanpy (BSD licence, https://github.com/scverse/scanpy).

Ported from ``scanpy.metrics._gearys_c`` as it stood in scanpy 1.11 rather than
imported at call time, because scanpy 1.12 changed the numba kernels here to
take the sparse graph as a single numba extension-type argument (via the new
``fast_array_utils`` dependency) instead of its raw ``data``/``indices``/
``indptr`` arrays. That dispatch path does not parallelise on this codebase's
CPU-only workloads: 15,000-signature permutation-null scoring measured ~4 ms
per signature (no speedup from 1 to 64 numba threads) versus ~0.24 ms per
signature for the plain-array kernels below (near-linear speedup with thread
count) -- a ~15-30x regression that dominated the Geary's C permutation-null
step. Vendoring the known-fast implementation decouples this hot path from
scanpy's internals entirely.

Only the ``graph (CSR) x dense (n_vars, n_cells) array`` path used by
``signature.py`` is exposed here; the AnnData-integration wrapper and the
sparse-vals path from the original ``scanpy.metrics.gearys_c`` are omitted
since scviva-tools never calls them.
"""

from __future__ import annotations

import threading
from functools import wraps
from typing import TYPE_CHECKING

import numba
import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Parallel/serial dispatch (mirrors scanpy's `_compat.njit`)
# ---------------------------------------------------------------------------
# A parallel numba function invoked from inside a `ThreadPoolExecutor` worker
# can deadlock under the `workqueue` threading layer. Compile both a parallel
# and a serial variant up front and pick one at call time based on the calling
# thread, exactly as scanpy 1.11 did.


def _numba_threading_layer() -> str | None:
    try:
        return numba.threading_layer()
    except Exception:  # noqa: BLE001 -- not yet initialised (no parallel call made)
        return None


def _is_in_unsafe_thread_pool() -> bool:
    current_thread = threading.current_thread()
    return current_thread.name.startswith(
        "ThreadPoolExecutor"
    ) and _numba_threading_layer() not in ("tbb", "omp")


def _njit(fn: Callable) -> Callable:
    """Jit-compile *fn*, dispatching to a parallel or sequential build at call time."""
    fns = {parallel: numba.njit(fn, cache=True, parallel=parallel) for parallel in (True, False)}

    @wraps(fn)
    def wrapper(*args, **kwargs):
        parallel = not _is_in_unsafe_thread_pool()
        return fns[parallel](*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
# There can be a fair amount of numerical instability here (big reductions),
# so data is cast to float64.


@numba.njit(cache=True, parallel=False)
def _gearys_c_inner_sparse_x_densevec(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
    w: np.float64,
) -> np.float64:
    x_bar = x.mean()
    total = 0.0
    n = len(x)
    for i in numba.prange(n):
        s = slice(g_indptr[i], g_indptr[i + 1])
        i_indices = g_indices[s]
        i_data = g_data[s]
        total += np.sum(i_data * ((x[i] - x[i_indices]) ** 2))
    numerator = (n - 1) * total
    denom = 2 * w * ((x - x_bar) ** 2).sum()
    return numerator / denom


@_njit
def _gearys_c_mtx(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    m, n = x.shape
    assert n == len(g_indptr) - 1
    w = g_data.sum()
    out = np.zeros(m, dtype=np.float64)
    for k in numba.prange(m):
        x_vec = x[k, :].astype(np.float64)
        out[k] = _gearys_c_inner_sparse_x_densevec(g_data, g_indices, g_indptr, x_vec, w)
    return out


def _gearys_c(graph: sparse.spmatrix, vals: np.ndarray) -> np.ndarray:
    """Geary's C for each row of *vals* against the graph *graph*.

    Parameters
    ----------
    graph : scipy.sparse matrix of shape (n_cells, n_cells)
        Cell-cell neighbor graph weight matrix.
    vals : np.ndarray of shape (n_vars, n_cells)
        Per-cell values to test for autocorrelation, one row per variable.

    Returns
    -------
    np.ndarray of shape (n_vars,)
        Geary's C per row of *vals*.
    """
    g = graph.tocsr() if sparse.issparse(graph) else sparse.csr_matrix(graph)
    return _gearys_c_mtx(g.data, g.indices, g.indptr, np.asarray(vals))
