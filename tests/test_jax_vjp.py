import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import magnus
import magnus.jax as magnus_jax


def test_jax_matrix_record_vjp_controls_return_contract():
    data = jnp.arange(36.0).reshape(9, 2, 2) / 30.0
    args = (4, lambda _: data, 0.0, 1.0, 9)

    out = magnus_jax.compute(*args, matrix_backend="Fixed2", integrator="Boole")
    recorded_out, carry = magnus_jax.compute(
        *args,
        matrix_backend="Fixed2",
        integrator="Boole",
        record_vjp=True,
    )

    assert out.shape == recorded_out.shape == (2, 2)
    assert carry.shape == (3, 3, 9, 2, 2)
    np.testing.assert_allclose(out, recorded_out, rtol=0.0, atol=0.0)


def test_jax_spacecurve_record_vjp_controls_return_contract():
    data = jnp.arange(27.0).reshape(9, 3) / 30.0
    args = (4, lambda _: data, 0.0, 1.0, 9)

    out = magnus_jax.compute_sc(*args, integrator="Boole")
    recorded_out, carry = magnus_jax.compute_sc(
        *args,
        integrator="Boole",
        record_vjp=True,
    )

    assert out.shape == recorded_out.shape == (3,)
    assert carry.shape == (3, 3, 9, 2, 2)
    np.testing.assert_allclose(out, recorded_out, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("op", ["one", "many", "sum"])
@pytest.mark.parametrize("record_vjp", [False, True])
def test_jax_matrix_gradient_matches_direct_vjp(op, record_vjp):
    samples = 9
    n = 4
    t = jnp.linspace(0.0, 1.0, samples)
    data = jnp.stack(
        (
            jnp.stack((0.2 + t, jnp.sin(t)), axis=-1),
            jnp.stack((-0.3 * t, 0.5 - t * t), axis=-1),
        ),
        axis=-2,
    )
    cotangent_shape = (n, 2, 2) if op == "many" else (2, 2)
    cotangent = jnp.linspace(0.2, 1.1, np.prod(cotangent_shape)).reshape(cotangent_shape)

    def loss(value):
        result = magnus_jax.compute(
            n,
            lambda _: value,
            0.0,
            1.0,
            samples,
            op=op,
            matrix_backend="Fixed2",
            integrator="Boole",
            record_vjp=record_vjp,
        )
        out = result[0] if record_vjp else result
        return jnp.sum(out * cotangent)

    actual = jax.jit(jax.grad(loss))(data)
    expected = magnus.compute_vjp(
        n,
        lambda _: np.asarray(data),
        np.asarray(cotangent),
        0.0,
        1.0,
        samples,
        op=op,
        matrix_backend="Fixed2",
        integrator="Boole",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op", ["one", "many", "sum"])
@pytest.mark.parametrize("record_vjp", [False, True])
def test_jax_spacecurve_gradient_matches_direct_vjp(op, record_vjp):
    samples = 9
    n = 4
    t = jnp.linspace(0.0, 1.0, samples)
    data = jnp.stack((jnp.sin(t) + 0.1, jnp.cos(0.7 * t), 0.2 + t * t), axis=-1)
    cotangent_shape = (n, 3) if op == "many" else (3,)
    cotangent = jnp.linspace(0.2, 1.1, np.prod(cotangent_shape)).reshape(cotangent_shape)

    def loss(value):
        result = magnus_jax.compute_sc(
            n,
            lambda _: value,
            0.0,
            1.0,
            samples,
            op=op,
            integrator="Boole",
            record_vjp=record_vjp,
        )
        out = result[0] if record_vjp else result
        return jnp.sum(out * cotangent)

    actual = jax.jit(jax.grad(loss))(data)
    expected = magnus.compute_sc_vjp(
        n,
        lambda _: np.asarray(data),
        np.asarray(cotangent),
        0.0,
        1.0,
        samples,
        op=op,
        integrator="Boole",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("record_vjp", [False, True])
def test_jax_complex_matrix_gradient_matches_bilinear_vjp(record_vjp):
    samples = 9
    n = 4
    t = jnp.linspace(0.0, 1.0, samples)
    real = jnp.stack(
        (
            jnp.stack((0.2 + t, jnp.sin(t)), axis=-1),
            jnp.stack((-0.3 * t, 0.5 - t * t), axis=-1),
        ),
        axis=-2,
    )
    imaginary = jnp.stack(
        (
            jnp.stack((0.1 * t, 0.2 - t), axis=-1),
            jnp.stack((0.3 + t * t, -0.4 * t), axis=-1),
        ),
        axis=-2,
    )
    data = real + 1j * imaginary
    cotangent = jnp.asarray([[0.2 - 0.1j, 0.4 + 0.3j], [0.7 - 0.2j, 1.1 + 0.4j]])

    def loss(value):
        result = magnus_jax.compute(
            n,
            lambda _: value,
            0.0,
            1.0,
            samples,
            op="sum",
            dtype=jnp.complex128,
            matrix_backend="Fixed2",
            integrator="Boole",
            record_vjp=record_vjp,
        )
        out = result[0] if record_vjp else result
        return jnp.real(jnp.sum(out * cotangent))

    actual = jax.jit(jax.grad(loss))(data)
    expected = magnus.compute_vjp(
        n,
        lambda _: np.asarray(data),
        np.asarray(cotangent),
        0.0,
        1.0,
        samples,
        op="sum",
        dtype=np.complex128,
        matrix_backend="Fixed2",
        integrator="Boole",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("record_vjp", [False, True])
def test_jax_complex_spacecurve_gradient_matches_bilinear_vjp(record_vjp):
    samples = 9
    n = 4
    t = jnp.linspace(0.0, 1.0, samples)
    data = jnp.stack(
        (
            jnp.sin(t) + 0.1j * t,
            jnp.cos(0.7 * t) + 1j * (0.2 - t),
            0.2 + t * t + 0.3j * jnp.sin(0.5 * t),
        ),
        axis=-1,
    )
    cotangent = jnp.asarray([0.2 - 0.1j, 0.5 + 0.3j, 1.1 - 0.4j])

    def loss(value):
        result = magnus_jax.compute_sc(
            n,
            lambda _: value,
            0.0,
            1.0,
            samples,
            op="sum",
            dtype=jnp.complex128,
            integrator="Boole",
            record_vjp=record_vjp,
        )
        out = result[0] if record_vjp else result
        return jnp.real(jnp.sum(out * cotangent))

    actual = jax.jit(jax.grad(loss))(data)
    expected = magnus.compute_sc_vjp(
        n,
        lambda _: np.asarray(data),
        np.asarray(cotangent),
        0.0,
        1.0,
        samples,
        op="sum",
        dtype=np.complex128,
        integrator="Boole",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("record_vjp", [False, True])
def test_jax_order_one_gradient_handles_empty_carry(record_vjp):
    data = jnp.arange(36.0).reshape(9, 2, 2) / 30.0
    cotangent = jnp.asarray([[0.2, 0.4], [0.7, 1.1]])

    def loss(value):
        result = magnus_jax.compute(
            1,
            lambda _: value,
            0.0,
            1.0,
            9,
            op="one",
            matrix_backend="Fixed2",
            integrator="Boole",
            record_vjp=record_vjp,
        )
        out = result[0] if record_vjp else result
        return jnp.sum(out * cotangent)

    actual = jax.jit(jax.grad(loss))(data)
    expected = magnus.compute_vjp(
        1,
        lambda _: np.asarray(data),
        np.asarray(cotangent),
        0.0,
        1.0,
        9,
        op="one",
        matrix_backend="Fixed2",
        integrator="Boole",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
