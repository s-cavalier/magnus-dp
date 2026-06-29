import operator
from typing import Any

import numpy as np

from . import _core

try:
    import jax
    import jax.numpy as jnp
except ImportError as err:
    raise ImportError("magnus.jax requires JAX to be installed") from err

__all__ = [
    "register_ffi_targets",
    "one_s",
    "many_s",
    "sum_s",
    "one_sc_s",
    "many_sc_s",
    "sum_sc_s",
]

_REGISTERED = False


def register_ffi_targets() -> None:
    global _REGISTERED

    if _REGISTERED:
        return

    if not hasattr(_core, "registrations"):
        raise RuntimeError(
            "magnus._core was not built with JAX FFI support. "
            "Rebuild with -DMAGNUS_ENABLE_JAX_FFI=ON."
        )

    for name, target in _core.registrations().items():
        jax.ffi.register_ffi_target(name, target, platform="cpu")

    _REGISTERED = True


def _matrix_output_type(op: str, n: int, data: Any):
    shape = tuple(data.shape)
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError("data must have shape (samples, dim, dim)")

    dim = shape[1]
    if op == "many":
        return jax.ShapeDtypeStruct((operator.index(n), dim, dim), data.dtype)

    return jax.ShapeDtypeStruct((dim, dim), data.dtype)


def _spacecurve_output_type(op: str, n: int, data: Any):
    shape = tuple(data.shape)
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError("data must have shape (samples, 3)")

    if op == "many":
        return jax.ShapeDtypeStruct((operator.index(n), 3), data.dtype)

    return jax.ShapeDtypeStruct((3,), data.dtype)


def _matrix_call(op: str, n: int, data: Any, t0: float, tf: float, matrix_backend: str, integrator: str):
    register_ffi_targets()
    data = jnp.asarray(data)
    call = jax.ffi.ffi_call("magnus_matrix", _matrix_output_type(op, n, data))
    return call(
        data,
        n=np.uint64(operator.index(n)),
        op=op,
        t0=float(t0),
        tf=float(tf),
        matrix_backend=matrix_backend,
        integrator=integrator,
    )


def _spacecurve_call(op: str, n: int, data: Any, t0: float, tf: float, integrator: str):
    register_ffi_targets()
    data = jnp.asarray(data)
    call = jax.ffi.ffi_call("magnus_spacecurve", _spacecurve_output_type(op, n, data))
    return call(
        data,
        n=np.uint64(operator.index(n)),
        op=op,
        t0=float(t0),
        tf=float(tf),
        integrator=integrator,
    )


def one_s(n: int, data: Any, t0: float, tf: float, matrix_backend: str = "Auto", integrator: str = "Auto"):
    return _matrix_call("one", n, data, t0, tf, matrix_backend, integrator)


def many_s(n: int, data: Any, t0: float, tf: float, matrix_backend: str = "Auto", integrator: str = "Auto"):
    return _matrix_call("many", n, data, t0, tf, matrix_backend, integrator)


def sum_s(n: int, data: Any, t0: float, tf: float, matrix_backend: str = "Auto", integrator: str = "Auto"):
    return _matrix_call("sum", n, data, t0, tf, matrix_backend, integrator)


def one_sc_s(n: int, data: Any, t0: float, tf: float, integrator: str = "Auto"):
    return _spacecurve_call("one", n, data, t0, tf, integrator)


def many_sc_s(n: int, data: Any, t0: float, tf: float, integrator: str = "Auto"):
    return _spacecurve_call("many", n, data, t0, tf, integrator)


def sum_sc_s(n: int, data: Any, t0: float, tf: float, integrator: str = "Auto"):
    return _spacecurve_call("sum", n, data, t0, tf, integrator)


register_ffi_targets()
