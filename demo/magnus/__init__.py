import operator
from typing import Callable

import numpy as np

from ._core import max_order
from ._core import _replace_gl_table
from ._core import many as many_s
from ._core import many_sc as many_sc_s
from ._core import one as one_s
from ._core import one_sc as one_sc_s
from ._core import sum as sum_s
from ._core import sum_sc as sum_sc_s

__all__ = [
    "max_order",
    "replace_gl_table",
    "one_s",
    "many_s",
    "sum_s",
    "one_sc_s",
    "many_sc_s",
    "sum_sc_s",
    "one",
    "many",
    "sum",
    "one_sc",
    "many_sc",
    "sum_sc",
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
    if sample_count < 2:
        raise ValueError("samples must be at least 2")

    t = np.linspace(t0, tf, sample_count)
    values = f(t) if vectorized else [f(float(ti)) for ti in t]
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype))

    if (
        data.ndim != 3
        or data.shape[0] != sample_count
        or data.shape[1] != data.shape[2]
    ):
        raise ValueError(
            "callable must return samples with shape (samples, dim, dim)"
        )

    return data


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
    if sample_count < 2:
        raise ValueError("samples must be at least 2")

    t = np.linspace(t0, tf, sample_count)
    values = f(t) if vectorized else [f(float(ti)) for ti in t]
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype))

    if data.ndim != 2 or data.shape != (sample_count, 3):
        raise ValueError(
            "callable must return samples with shape (samples, 3)"
        )

    return data


def one(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    data = _sample_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return one_s(n, data, t0, tf, matrix_backend=matrix_backend, integrator=integrator)


def many(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    data = _sample_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return many_s(n, data, t0, tf, matrix_backend=matrix_backend, integrator=integrator)


def sum(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    data = _sample_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return sum_s(n, data, t0, tf, matrix_backend=matrix_backend, integrator=integrator)


def one_sc(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> np.ndarray:
    data = _sample_spacecurve_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return one_sc_s(n, data, t0, tf, integrator=integrator)


def many_sc(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> np.ndarray:
    data = _sample_spacecurve_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return many_sc_s(n, data, t0, tf, integrator=integrator)


def sum_sc(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype=None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> np.ndarray:
    data = _sample_spacecurve_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return sum_sc_s(n, data, t0, tf, integrator=integrator)
