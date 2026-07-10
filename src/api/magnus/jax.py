import operator
from functools import partial
from typing import Any, Callable

import numpy as np

from . import _core

try:
    import jax
    import jax.numpy as jnp
except ImportError as err:
    raise ImportError("magnus.jax requires JAX to be installed") from err

__all__ = [
    "register_ffi_targets",
    "compute",
    "compute_sc",
]

_REGISTERED = False


def register_ffi_targets() -> None:
    global _REGISTERED

    if _REGISTERED: return

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
    if len(shape) < 2: raise ValueError("cannot infer matrix output shape from data")

    dim = shape[1]
    if op == "many": return jax.ShapeDtypeStruct((operator.index(n), dim, dim), data.dtype)

    return jax.ShapeDtypeStruct((dim, dim), data.dtype)


def _spacecurve_output_type(op: str, n: int, data: Any):
    if op == "many": return jax.ShapeDtypeStruct((operator.index(n), 3), data.dtype)

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


def _matrix_fwd(
    data: Any,
    op: str,
    n: int,
    t0: float,
    tf: float,
    matrix_backend: str,
    integrator: str,
):
    data = jnp.asarray(data)
    out = _matrix_call(op, n, data, t0, tf, matrix_backend, integrator)
    return out, (data,)


def _spacecurve_fwd(
    data: Any,
    op: str,
    n: int,
    t0: float,
    tf: float,
    integrator: str,
):
    data = jnp.asarray(data)
    out = _spacecurve_call(op, n, data, t0, tf, integrator)
    return out, (data,)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def _matrix_call_vjp(
    data: Any,
    op: str,
    n: int,
    t0: float,
    tf: float,
    matrix_backend: str,
    integrator: str,
):
    return _matrix_call(op, n, data, t0, tf, matrix_backend, integrator)


def _matrix_bwd(
    op: str,
    n: int,
    t0: float,
    tf: float,
    matrix_backend: str,
    integrator: str,
    residuals: tuple[Any, ...],
    cotangent: Any,
):
    del op, n, t0, tf, matrix_backend, integrator, residuals, cotangent
    raise NotImplementedError(
        "magnus.jax reverse-mode gradients need the C++ VJP FFI target, "
        "which has not been implemented yet."
    )


_matrix_call_vjp.defvjp(_matrix_fwd, _matrix_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def _spacecurve_call_vjp(
    data: Any,
    op: str,
    n: int,
    t0: float,
    tf: float,
    integrator: str,
):
    return _spacecurve_call(op, n, data, t0, tf, integrator)


def _spacecurve_bwd(
    op: str,
    n: int,
    t0: float,
    tf: float,
    integrator: str,
    residuals: tuple[Any, ...],
    cotangent: Any,
):
    del op, n, t0, tf, integrator, residuals, cotangent
    raise NotImplementedError(
        "magnus.jax reverse-mode gradients need the C++ VJP FFI target, "
        "which has not been implemented yet."
    )


_spacecurve_call_vjp.defvjp(_spacecurve_fwd, _spacecurve_bwd)


def _sample_callable(
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
):
    sample_count = operator.index(samples)
    t = jnp.linspace(t0, tf, sample_count)
    values = f(t) if vectorized else jax.vmap(f)(t)
    return jnp.asarray(values, dtype=dtype)


def _sample_spacecurve_callable(
    f: Callable,
    t0: float,
    tf: float,
    samples: int,
    *,
    dtype: Any = None,
    vectorized: bool = True,
):
    sample_count = operator.index(samples)
    t = jnp.linspace(t0, tf, sample_count)
    values = f(t) if vectorized else jax.vmap(f)(t)
    return jnp.asarray(values, dtype=dtype)


def _compute_sampled(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    *,
    op: str = "sum",
    matrix_backend: str = "Auto",
    integrator: str = "Auto",
):
    return _matrix_call_vjp(data, op, n, t0, tf, matrix_backend, integrator)


def _compute_sc_sampled(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    *,
    op: str = "sum",
    integrator: str = "Auto",
):
    return _spacecurve_call_vjp(data, op, n, t0, tf, integrator)


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
):
    data = _sample_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return _compute_sampled(
        n,
        data,
        t0,
        tf,
        op=op,
        matrix_backend=matrix_backend,
        integrator=integrator,
    )


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
):
    data = _sample_spacecurve_callable(
        f,
        t0,
        tf,
        samples,
        dtype=dtype,
        vectorized=vectorized,
    )
    return _compute_sc_sampled(
        n,
        data,
        t0,
        tf,
        op=op,
        integrator=integrator,
    )


register_ffi_targets()
