import operator
from typing import Callable

import numpy as np

from ._core import max_order
from ._core import many as many_s
from ._core import one as one_s
from ._core import sum as sum_s

__all__ = [
    "max_order",
    "one_s",
    "many_s",
    "sum_s",
    "one",
    "many",
    "sum",
]


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
