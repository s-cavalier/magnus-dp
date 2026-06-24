import numpy as np
from typing import Any, Callable

def max_order() -> int: 
    """Specifies the current max Legendre-Gauss degree available."""

def one_s(
    n: int,
    data: np.ndarray,
    t0: float,
    tf: float,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute magnus expansion order n using discretizations on data. <br>
    n is magnus order, data should be an array of evenly-spaced samples from a matrix function.
    t0 is the start time, tf is the end time.
    """

def many_s(
    n: int,
    data: np.ndarray,
    t0: float,
    tf: float,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute magnus expansion order 1 through n, reusing intermediary data. <br>
    n is magnus order, data should be an array of evenly-spaced samples from a matrix function.
    t0 is the start time, tf is the end time.

    """

def sum_s(
    n: int,
    data: np.ndarray,
    t0: float,
    tf: float,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute the sum of Magnus expansion orders 1 through n. <br>
    n is magnus order, data should be an array of evenly-spaced samples from a matrix function.
    t0 is the start time, tf is the end time.
    """

def one(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute magnus expansion order n by sampling f on evenly-spaced points.
    """

def many(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute magnus expansion orders 1 through n by sampling f on evenly-spaced points.
    """

def sum(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute the sum of Magnus expansion orders 1 through n by sampling f on evenly-spaced points.
    """
