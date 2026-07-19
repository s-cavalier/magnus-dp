from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable

import numpy as np

from ._core import _replace_gl_table
from ._core import compute as _compute_sampled
from ._core import compute_sc as _compute_sc_sampled
from ._core import compute_sc_vjp as _compute_sc_vjp_sampled
from ._core import compute_vjp as _compute_vjp_sampled
from ._core import gl_backends
from ._core import integrators
from ._core import matrix_backends
from ._core import max_order
from ._core import numeric_backends
from ._core import ops

if TYPE_CHECKING:
    from ._generated_typing import GLBackendName, IntegratorName, KernelOpName, MatrixBackendName

__all__ = [
    "max_order",
    "numeric_backends",
    "matrix_backends",
    "integrators",
    "gl_backends",
    "ops",
    "replace_gl_table",
    "compute",
    "compute_sc",
    "compute_vjp",
    "compute_sc_vjp",
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the Magnus output and, when requested, its saved forward carry."""
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
        gl_backend=gl_backend,
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the SpaceCurve Magnus output and, when requested, its saved forward carry."""
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
        gl_backend=gl_backend,
        record_vjp=record_vjp,
    )


def compute_vjp(
    n: int,
    f: Callable,
    cotangent,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype=None,
    vectorized: bool = True,
    matrix_backend: MatrixBackendName = "Auto",
    integrator: IntegratorName = "Auto",
    vjp_data=None,
) -> np.ndarray:
    data = _sample_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    cotangent = np.ascontiguousarray(np.asarray(cotangent, dtype=data.dtype))
    return _compute_vjp_sampled(
        n,
        data,
        cotangent,
        vjp_data=vjp_data,
        t0=t0,
        tf=tf,
        op=op,
        matrix_backend=matrix_backend,
        integrator=integrator,
    )


def compute_sc_vjp(
    n: int,
    f: Callable,
    cotangent,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype=None,
    vectorized: bool = True,
    integrator: IntegratorName = "Auto",
    vjp_data=None,
) -> np.ndarray:
    data = _sample_spacecurve_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    cotangent = np.ascontiguousarray(np.asarray(cotangent, dtype=data.dtype))
    return _compute_sc_vjp_sampled(
        n,
        data,
        cotangent,
        vjp_data=vjp_data,
        t0=t0,
        tf=tf,
        op=op,
        integrator=integrator,
    )
