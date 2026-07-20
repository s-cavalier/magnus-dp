from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from . import _core

if TYPE_CHECKING:
    from ._generated_typing import GLBackendName, IntegratorName, KernelOpName, MatrixBackendName

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


def _matrix_output_type(op: KernelOpName, n: int, data: Any):
    shape = tuple(data.shape)
    dim = shape[1]
    if op == "many": return jax.ShapeDtypeStruct((operator.index(n), dim, dim), data.dtype)

    return jax.ShapeDtypeStruct((dim, dim), data.dtype)


def _matrix_carry_type(n: int, data: Any):
    shape = tuple(data.shape)
    return jax.ShapeDtypeStruct(((operator.index(n) + 3) // 2, operator.index(n) - 1, shape[0], shape[1], shape[2]), data.dtype)


def _spacecurve_output_type(op: KernelOpName, n: int, data: Any):
    if op == "many": return jax.ShapeDtypeStruct((operator.index(n), 3), data.dtype)

    return jax.ShapeDtypeStruct((3,), data.dtype)


def _spacecurve_carry_type(n: int, data: Any):
    shape = tuple(data.shape)
    return jax.ShapeDtypeStruct(((operator.index(n) + 3) // 2, operator.index(n) - 1, shape[0], 2, 2), data.dtype)


def _matrix_call(
    op: KernelOpName,
    n: int,
    data: Any,
    t0: float,
    tf: float,
    matrix_backend: MatrixBackendName,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
):
    register_ffi_targets()
    data = jnp.asarray(data)
    if record_vjp:
        call = jax.ffi.ffi_call(
            "magnus_matrix_fwd",
            (
                _matrix_output_type(op, n, data),
                _matrix_carry_type(n, data),
            ),
        )
        return call(
            data,
            n=np.uint64(operator.index(n)),
            op=op,
            t0=float(t0),
            tf=float(tf),
            matrix_backend=matrix_backend,
            integrator=integrator,
            gl_backend=gl_backend,
        )

    call = jax.ffi.ffi_call("magnus_matrix", _matrix_output_type(op, n, data))
    return call(
        data,
        n=np.uint64(operator.index(n)),
        op=op,
        t0=float(t0),
        tf=float(tf),
        matrix_backend=matrix_backend,
        integrator=integrator,
        gl_backend=gl_backend,
    )


def _spacecurve_call(
    op: KernelOpName,
    n: int,
    data: Any,
    t0: float,
    tf: float,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
):
    register_ffi_targets()
    data = jnp.asarray(data)
    if record_vjp:
        call = jax.ffi.ffi_call(
            "magnus_spacecurve_fwd",
            (
                _spacecurve_output_type(op, n, data),
                _spacecurve_carry_type(n, data),
            ),
        )
        return call(
            data,
            n=np.uint64(operator.index(n)),
            op=op,
            t0=float(t0),
            tf=float(tf),
            integrator=integrator,
            gl_backend=gl_backend,
        )

    call = jax.ffi.ffi_call("magnus_spacecurve", _spacecurve_output_type(op, n, data))
    return call(
        data,
        n=np.uint64(operator.index(n)),
        op=op,
        t0=float(t0),
        tf=float(tf),
        integrator=integrator,
        gl_backend=gl_backend,
    )


def _matrix_bwd_call(
    op: KernelOpName,
    n: int,
    data: Any,
    cotangent: Any,
    carry: Any | None,
    t0: float,
    tf: float,
    matrix_backend: MatrixBackendName,
    integrator: IntegratorName,
):
    call = jax.ffi.ffi_call(
        "magnus_matrix_bwd",
        jax.ShapeDtypeStruct(data.shape, data.dtype),
    )
    args = (data, cotangent) if carry is None else (data, cotangent, carry)
    return call(
        *args,
        n=np.uint64(operator.index(n)),
        op=op,
        t0=float(t0),
        tf=float(tf),
        matrix_backend=matrix_backend,
        integrator=integrator,
    )


def _spacecurve_bwd_call(
    op: KernelOpName,
    n: int,
    data: Any,
    cotangent: Any,
    carry: Any | None,
    t0: float,
    tf: float,
    integrator: IntegratorName,
):
    call = jax.ffi.ffi_call(
        "magnus_spacecurve_bwd",
        jax.ShapeDtypeStruct(data.shape, data.dtype),
    )
    args = (data, cotangent) if carry is None else (data, cotangent, carry)
    return call(
        *args,
        n=np.uint64(operator.index(n)),
        op=op,
        t0=float(t0),
        tf=float(tf),
        integrator=integrator,
    )


def _matrix_fwd(
    data: Any,
    op: KernelOpName,
    n: int,
    t0: float,
    tf: float,
    matrix_backend: MatrixBackendName,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
):
    data = jnp.asarray(data)
    if record_vjp:
        out, carry = _matrix_call(op, n, data, t0, tf, matrix_backend, integrator, gl_backend, True)
        return (out, jax.lax.stop_gradient(carry)), (data, carry)

    out = _matrix_call(op, n, data, t0, tf, matrix_backend, integrator, gl_backend, False)
    return out, (data,)


def _spacecurve_fwd(
    data: Any,
    op: KernelOpName,
    n: int,
    t0: float,
    tf: float,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
):
    data = jnp.asarray(data)
    if record_vjp:
        out, carry = _spacecurve_call(op, n, data, t0, tf, integrator, gl_backend, True)
        return (out, jax.lax.stop_gradient(carry)), (data, carry)

    out = _spacecurve_call(op, n, data, t0, tf, integrator, gl_backend, False)
    return out, (data,)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def _matrix_custom(
    data: Any,
    op: KernelOpName,
    n: int,
    t0: float,
    tf: float,
    matrix_backend: MatrixBackendName,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
):
    return _matrix_call(op, n, data, t0, tf, matrix_backend, integrator, gl_backend, record_vjp)


def _matrix_bwd(
    op: KernelOpName,
    n: int,
    t0: float,
    tf: float,
    matrix_backend: MatrixBackendName,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
    residuals: tuple[Any, ...],
    cotangent: Any,
):
    if record_vjp:
        data, carry = residuals
        cotangent = cotangent[0]
    else:
        (data,) = residuals
        carry = None
    dA = _matrix_bwd_call(
        op,
        n,
        data,
        cotangent,
        carry,
        t0,
        tf,
        matrix_backend,
        integrator,
    )
    return (dA,)


_matrix_custom.defvjp(_matrix_fwd, _matrix_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def _spacecurve_custom(
    data: Any,
    op: KernelOpName,
    n: int,
    t0: float,
    tf: float,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
):
    return _spacecurve_call(op, n, data, t0, tf, integrator, gl_backend, record_vjp)


def _spacecurve_bwd(
    op: KernelOpName,
    n: int,
    t0: float,
    tf: float,
    integrator: IntegratorName,
    gl_backend: GLBackendName,
    record_vjp: bool,
    residuals: tuple[Any, ...],
    cotangent: Any,
):
    if record_vjp:
        data, carry = residuals
        cotangent = cotangent[0]
    else:
        (data,) = residuals
        carry = None
    dA = _spacecurve_bwd_call(
        op,
        n,
        data,
        cotangent,
        carry,
        t0,
        tf,
        integrator,
    )
    return (dA,)


_spacecurve_custom.defvjp(_spacecurve_fwd, _spacecurve_bwd)


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
    op: KernelOpName = "sum",
    matrix_backend: MatrixBackendName = "Auto",
    integrator: IntegratorName = "Auto",
    gl_backend: GLBackendName = "Auto",
    record_vjp: bool = False,
):
    return _matrix_custom(data, op, n, t0, tf, matrix_backend, integrator, gl_backend, record_vjp)


def _compute_sc_sampled(
    n: int,
    data: Any,
    t0: float,
    tf: float,
    *,
    op: KernelOpName = "sum",
    integrator: IntegratorName = "Auto",
    gl_backend: GLBackendName = "Auto",
    record_vjp: bool = False,
):
    return _spacecurve_custom(data, op, n, t0, tf, integrator, gl_backend, record_vjp)


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
    record_vjp: bool = False,
):
    """Return the Magnus output and, when requested, its saved forward carry."""
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
        gl_backend=gl_backend,
        record_vjp=record_vjp,
    )


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
    record_vjp: bool = False,
):
    """Return the SpaceCurve Magnus output and, when requested, its saved forward carry."""
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
        gl_backend=gl_backend,
        record_vjp=record_vjp,
    )


register_ffi_targets()
