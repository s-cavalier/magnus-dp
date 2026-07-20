import numpy as np
from typing import Any, Callable, Literal, overload

from ._generated_typing import GLBackendName, IntegratorName, KernelOpName, MatrixBackendName, NumericBackendName

def max_order() -> int:
    """Specifies the current max Legendre-Gauss degree available."""

def numeric_backends() -> tuple[NumericBackendName, ...]: ...

def matrix_backends() -> tuple[MatrixBackendName, ...]: ...

def integrators() -> tuple[IntegratorName, ...]: ...

def gl_backends() -> tuple[GLBackendName, ...]: ...

def ops() -> tuple[KernelOpName, ...]: ...

def replace_gl_table(max_order: int) -> None:
    """
    Generate and install a new Gauss-Legendre table through max_order.
    """

@overload
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: Literal[False] = False,
) -> np.ndarray:
    """
    Compute a Magnus operation by sampling an R -> matrix function.
    """

@overload
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...

@overload
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

@overload
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: Literal[False] = False,
) -> np.ndarray:
    """
    Compute a SpaceCurve Magnus operation by sampling an R -> R^3 function.
    """

@overload
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...

@overload
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
    gl_backend: GLBackendName = "Auto",
    record_vjp: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

def compute_vjp(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    cotangent: np.ndarray,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: MatrixBackendName = "Auto",
    integrator: IntegratorName = "Auto",
    vjp_data: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the VJP with respect to sampled matrix data."""

def compute_sc_vjp(
    n: int,
    f: Callable[[np.ndarray], np.ndarray],
    cotangent: np.ndarray,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: KernelOpName = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    integrator: IntegratorName = "Auto",
    vjp_data: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the VJP with respect to sampled SpaceCurve data."""
