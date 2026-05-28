import numpy as np

def max_order() -> int: 
    """Specifies the current max Legendre-Gauss degree available."""

def one(n: int, data: np.ndarray, t0: float, tf: float) -> np.ndarray:
    """
    Compute magnus expansion order n using discretizations on data. <br>
    n is magnus order, data should be an array of evenly-spaced samples from a matrix function.
    t0 is the start time, tf is the end time.
    """

def many(n: int, data: np.ndarray, t0: float, tf: float) -> np.ndarray:
    """
    Compute magnus expansion order 1 through n, reusing intermediary data. <br>
    n is magnus order, data should be an array of evenly-spaced samples from a matrix function.
    t0 is the start time, tf is the end time.

    """

def sum(n: int, data: np.ndarray, t0: float, tf: float) -> np.ndarray:
    """
    Compute the sum of Magnus expansion orders 1 through n. <br>
    n is magnus order, data should be an array of evenly-spaced samples from a matrix function.
    t0 is the start time, tf is the end time.
    """
