import numpy as np

def max_order() -> int: 
    """Specifies the current max Legendre-Gauss degree available."""

def one(n: int, data: np.ndarray, t0: float, tf: float) -> np.ndarray:
    """Compute magnus expansion order n using discretizations on data."""

def many(n: int, data: np.ndarray, t0: float, tf: float) -> np.ndarray:
    """Compute magnus expansion order 1 through n, reusing intermediary data."""

def sum(n: int, data: np.ndarray, t0: float, tf: float) -> np.ndarray:
    """Compute the sum of Magnus expansion orders 1 through n."""
