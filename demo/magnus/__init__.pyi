import numpy as np
from typing import Any, Callable

def max_order() -> int: 
    """Specifies the current max Legendre-Gauss degree available."""

def replace_gl_table(max_order: int) -> None:
    """
    Generate and install a new Gauss-Legendre table through max_order.
    """

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

def one_sc_s(
    n: int,
    data: np.ndarray,
    t0: float,
    tf: float,
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute SpaceCurve Magnus expansion order n from 3-vector samples.
    """

def many_sc_s(
    n: int,
    data: np.ndarray,
    t0: float,
    tf: float,
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute SpaceCurve Magnus expansion orders 1 through n from 3-vector samples.
    """

def sum_sc_s(
    n: int,
    data: np.ndarray,
    t0: float,
    tf: float,
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute the sum of SpaceCurve Magnus expansion orders from 3-vector samples.
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

def one_sc(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute SpaceCurve Magnus expansion order n by sampling an R -> R^3 function.
    """

def many_sc(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute SpaceCurve Magnus expansion orders 1 through n by sampling an R -> R^3 function.
    """

def sum_sc(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> np.ndarray:
    """
    Compute the SpaceCurve Magnus expansion sum by sampling an R -> R^3 function.
    """
