from typing import Any

def register_ffi_targets() -> None: ...

def one_s(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> Any: ...

def many_s(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> Any: ...

def sum_s(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
) -> Any: ...

def one_sc_s(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    integrator: str = "Auto",
) -> Any: ...

def many_sc_s(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    integrator: str = "Auto",
) -> Any: ...

def sum_sc_s(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    integrator: str = "Auto",
) -> Any: ...
