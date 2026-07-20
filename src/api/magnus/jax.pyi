from typing import Any, Callable, Literal, overload

from ._generated_typing import GLBackendName, IntegratorName, KernelOpName, MatrixBackendName

def register_ffi_targets() -> None: ...

@overload
def compute(
    n: int,
    f: Callable,
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
) -> Any: ...

@overload
def compute(
    n: int,
    f: Callable,
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
) -> tuple[Any, Any]: ...

@overload
def compute(
    n: int,
    f: Callable,
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
) -> Any | tuple[Any, Any]: ...

@overload
def compute_sc(
    n: int,
    f: Callable,
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
) -> Any: ...

@overload
def compute_sc(
    n: int,
    f: Callable,
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
) -> tuple[Any, Any]: ...

@overload
def compute_sc(
    n: int,
    f: Callable,
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
) -> Any | tuple[Any, Any]: ...
