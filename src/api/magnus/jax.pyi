from typing import Any, Callable

def register_ffi_targets() -> None: ...

def compute(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: str = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> Any: ...

def compute_sc(
    n: int,
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    op: str = "sum",
    dtype: Any = None,
    vectorized: bool = True,
    integrator: str = "Auto",
) -> Any: ...
