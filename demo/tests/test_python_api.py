from pathlib import Path
import sys

import numpy as np
import pytest


DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import magnus


def matrix_values(t, dtype=np.float64):
    t = np.asarray(t)
    out = np.empty((t.size, 2, 2), dtype=dtype)
    out[:, 0, 0] = 1.0 + t
    out[:, 0, 1] = t * t
    out[:, 1, 0] = -0.5 * t
    out[:, 1, 1] = 2.0 - t
    return out


def matrix_value(t, dtype=np.float64):
    return matrix_values(np.asarray([t]), dtype=dtype)[0]


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_sampled_api_exports_core_entrypoints():
    assert set(magnus.__all__) == {
        "max_order",
        "one_s",
        "many_s",
        "sum_s",
        "one",
        "many",
        "sum",
    }
    assert callable(magnus.one_s)
    assert callable(magnus.many_s)
    assert callable(magnus.sum_s)


def test_vectorized_callable_api_matches_sampled_api():
    t0 = -0.25
    tf = 1.25
    samples = 9
    n = 3
    seen = []

    def f(t):
        seen.append(np.array(t, copy=True))
        return matrix_values(t)

    t = np.linspace(t0, tf, samples)
    data = matrix_values(t)

    one = magnus.one(
        n,
        f,
        t0,
        tf,
        samples,
        matrix_backend="Manual",
        integrator="Riemann",
    )
    many = magnus.many(
        n,
        f,
        t0,
        tf,
        samples,
        matrix_backend="Manual",
        integrator="Riemann",
    )
    total = magnus.sum(
        n,
        f,
        t0,
        tf,
        samples,
        matrix_backend="Manual",
        integrator="Riemann",
    )

    assert len(seen) == 3
    for values in seen:
        np.testing.assert_array_equal(values, t)

    assert_close(one, magnus.one_s(n, data, t0, tf, "Manual", "Riemann"))
    assert_close(many, magnus.many_s(n, data, t0, tf, "Manual", "Riemann"))
    assert_close(total, magnus.sum_s(n, data, t0, tf, "Manual", "Riemann"))


def test_pointwise_callable_api_matches_sampled_api():
    t0 = 0.0
    tf = 2.0
    samples = 9
    n = 2
    seen = []

    def f(t):
        assert isinstance(t, float)
        seen.append(t)
        return matrix_value(t)

    t = np.linspace(t0, tf, samples)
    data = matrix_values(t)

    actual = magnus.sum(
        n,
        f,
        t0,
        tf,
        samples,
        vectorized=False,
        matrix_backend="Fixed2",
        integrator="Trapezoid",
    )
    expected = magnus.sum_s(n, data, t0, tf, "Fixed2", "Trapezoid")

    np.testing.assert_array_equal(seen, t)
    assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_callable_dtype_controls_sample_dtype(dtype):
    def f(t):
        data = matrix_values(t, dtype=np.float64)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            data = data + 0.25j * data
        return data

    actual = magnus.one(
        1,
        f,
        0.0,
        1.0,
        9,
        dtype=dtype,
        matrix_backend="Manual",
        integrator="Riemann",
    )

    assert actual.dtype == np.dtype(dtype)

    data = np.asarray(f(np.linspace(0.0, 1.0, 9)), dtype=dtype)
    expected = magnus.one_s(1, data, 0.0, 1.0, "Manual", "Riemann")
    np.testing.assert_allclose(actual, expected)


def test_callable_api_preserves_operation_shapes():
    def f(t):
        return matrix_values(t)

    one = magnus.one(4, f, 0.0, 1.0, 9, integrator="Boole")
    many = magnus.many(4, f, 0.0, 1.0, 9, integrator="Boole")
    total = magnus.sum(4, f, 0.0, 1.0, 9, integrator="Boole")

    assert one.shape == (2, 2)
    assert many.shape == (4, 2, 2)
    assert total.shape == (2, 2)
    np.testing.assert_allclose(one, many[3], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(total, np.sum(many, axis=0), rtol=1e-11, atol=1e-12)


def test_callable_dispatch_kwargs_are_forwarded():
    def f(t):
        return np.zeros((np.asarray(t).size, 3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="invalid dispatch option"):
        magnus.one(1, f, 0.0, 1.0, 9, matrix_backend="Fixed2")

    with pytest.raises(ValueError, match="dispatch name"):
        magnus.one(1, f, 0.0, 1.0, 9, matrix_backend="Missing")

    with pytest.raises(ValueError, match="dispatch name"):
        magnus.one(1, f, 0.0, 1.0, 9, integrator="Missing")


def test_callable_sample_count_is_validated_before_calling_function():
    called = False

    def f(t):
        nonlocal called
        called = True
        return matrix_values(t)

    with pytest.raises(ValueError, match="samples must be at least 2"):
        magnus.one(1, f, 0.0, 1.0, 1)

    assert not called


@pytest.mark.parametrize(
    "bad_result",
    [
        np.zeros((9, 2), dtype=np.float64),
        np.zeros((8, 2, 2), dtype=np.float64),
        np.zeros((9, 2, 3), dtype=np.float64),
    ],
)
def test_callable_output_shape_is_validated(bad_result):
    def f(t):
        return bad_result

    with pytest.raises(ValueError, match="shape"):
        magnus.one(1, f, 0.0, 1.0, 9)


def test_sampled_api_still_validates_core_inputs():
    data = matrix_values(np.linspace(0.0, 1.0, 9))

    with pytest.raises(ValueError, match="n must be at least 1"):
        magnus.one_s(0, data, 0.0, 1.0)

    with pytest.raises(TypeError, match="supports dtypes"):
        magnus.one_s(1, data.astype(np.int64), 0.0, 1.0)
