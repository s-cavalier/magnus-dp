from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable

import numpy as np

from ._core import _replace_gl_table
from ._core import compute as _compute_sampled
from ._core import compute_sc as _compute_sc_sampled
from ._core import integrators
from ._core import matrix_backends
from ._core import max_order
from ._core import numeric_backends
from ._core import ops

if TYPE_CHECKING:
    from ._generated_typing import IntegratorName, KernelOpName, MatrixBackendName

__all__ = [
    "max_order",
    "numeric_backends",
    "matrix_backends",
    "integrators",
    "ops",
    "replace_gl_table",
    "compute",
    "compute_sc",
]


def replace_gl_table(max_order: int) -> None:
    table_order = operator.index(max_order)
    if table_order < 1:
        raise ValueError("max_order must be at least 1")

    total_size = table_order * (table_order + 1) // 2
    weights = np.empty(total_size, dtype=np.float64)
    nodes = np.empty(total_size, dtype=np.float64)

    offset = 0
    for order in range(1, table_order + 1):
        x, w = np.polynomial.legendre.leggauss(order)
        next_offset = offset + order
        nodes[offset:next_offset] = (x + 1) / 2
        weights[offset:next_offset] = w / 2
        offset = next_offset

    _replace_gl_table(table_order, weights, nodes)


def _sample_callable(
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
) -> np.ndarray:
    sample_count = operator.index(samples)
    t = np.linspace(t0, tf, sample_count)
    values = f(t) if vectorized else [f(float(ti)) for ti in t]
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def _sample_spacecurve_callable(
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
) -> np.ndarray:
    sample_count = operator.index(samples)
    t = np.linspace(t0, tf, sample_count)
    values = f(t) if vectorized else [f(float(ti)) for ti in t]
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def compute(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype=None,
    vectorized: bool = True,
    matrix_backend: MatrixBackendName = "Auto",
    integrator: IntegratorName = "Auto",
    record_vjp: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    data = _sample_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return _compute_sampled(
        n,
        data,
        t0,
        tf,
        op=op,
        matrix_backend=matrix_backend,
        integrator=integrator,
        record_vjp=record_vjp,
    )


def compute_sc(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype=None,
    vectorized: bool = True,
    integrator: IntegratorName = "Auto",
    record_vjp: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    data = _sample_spacecurve_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return _compute_sc_sampled(
        n,
        data,
        t0,
        tf,
        op=op,
        integrator=integrator,
        record_vjp=record_vjp,
    )
