import numpy as np
from typing import Any, Callable

from ._generated_typing import IntegratorName, KernelOpName, MatrixBackendName, NumericBackendName

def max_order() -> int:
    """Specifies the current max Legendre-Gauss degree available."""

def numeric_backends() -> tuple[NumericBackendName, ...]: ...

def matrix_backends() -> tuple[MatrixBackendName, ...]: ...

def integrators() -> tuple[IntegratorName, ...]: ...

def ops() -> tuple[KernelOpName, ...]: ...

def replace_gl_table(max_order: int) -> None:
    """
    Generate and install a new Gauss-Legendre table through max_order.
    """

def compute(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: MatrixBackendName = "Auto",
    integrator: IntegratorName = "Auto",
) -> np.ndarray:
    """
    Compute a Magnus operation by sampling an R -> matrix function.
    """

def compute_sc(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    integrator: IntegratorName = "Auto",
) -> np.ndarray:
    """
    Compute a SpaceCurve Magnus operation by sampling an R -> R^3 function.
    """
