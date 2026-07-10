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


def spacecurve_values(t, dtype=np.float64):
    t = np.asarray(t)
    out = np.empty((t.size, 3), dtype=dtype)
    out[:, 0] = np.sin(1.3 * t) + 0.2 * t
    out[:, 1] = np.cos(0.7 * t) - 0.1
    out[:, 2] = 0.4 + t * t
    return out


def spacecurve_value(t, dtype=np.float64):
    return spacecurve_values(np.asarray([t]), dtype=dtype)[0]


def test_public_api_exports_only_compute_entrypoints():
    assert set(magnus.__all__) == {
        "max_order",
        "numeric_backends",
        "matrix_backends",
        "integrators",
        "ops",
        "replace_gl_table",
        "compute",
        "compute_sc",
    }
    assert callable(magnus.replace_gl_table)
    assert callable(magnus.compute)
    assert callable(magnus.compute_sc)

    for removed in [
        "one_s",
        "many_s",
        "sum_s",
        "one_sc_s",
        "many_sc_s",
        "sum_sc_s",
        "one",
        "many",
        "sum",
        "one_sc",
        "many_sc",
        "sum_sc",
    ]:
        assert not hasattr(magnus, removed)


def test_dispatch_metadata_is_discoverable():
    assert magnus.ops() == ("one", "many", "sum")
    assert "Manual" in magnus.matrix_backends()
    assert "SpaceCurve" in magnus.matrix_backends()
    assert "Auto" in magnus.matrix_backends()
    assert "Riemann" in magnus.integrators()
    assert "Auto" in magnus.integrators()
    assert "float" in magnus.numeric_backends()
    assert "double" in magnus.numeric_backends()
    assert "Auto" not in magnus.numeric_backends()


def test_matrix_compute_ops_and_shapes():
    def f(t):
        return matrix_values(t)

    one = magnus.compute(4, f, 0.0, 1.0, 9, op="one", integrator="Boole")
    many = magnus.compute(4, f, 0.0, 1.0, 9, op="many", integrator="Boole")
    total = magnus.compute(4, f, 0.0, 1.0, 9, op="sum", integrator="Boole")

    assert one.shape == (2, 2)
    assert many.shape == (4, 2, 2)
    assert total.shape == (2, 2)
    np.testing.assert_allclose(one, many[3], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(total, np.sum(many, axis=0), rtol=1e-11, atol=1e-12)


def test_matrix_vectorized_and_pointwise_sampling_match():
    t0 = 0.0
    tf = 2.0
    samples = 9
    n = 2
    seen = []

    def pointwise(t):
        assert isinstance(t, float)
        seen.append(t)
        return matrix_value(t)

    vectorized = magnus.compute(
        n,
        matrix_values,
        t0,
        tf,
        samples,
        op="sum",
        matrix_backend="Fixed2",
        integrator="Trapezoid",
    )
    point = magnus.compute(
        n,
        pointwise,
        t0,
        tf,
        samples,
        op="sum",
        vectorized=False,
        matrix_backend="Fixed2",
        integrator="Trapezoid",
    )

    np.testing.assert_array_equal(seen, np.linspace(t0, tf, samples))
    np.testing.assert_allclose(point, vectorized, rtol=1e-11, atol=1e-12)


def test_spacecurve_compute_ops_and_shapes():
    def f(t):
        return spacecurve_values(t)

    one = magnus.compute_sc(4, f, 0.0, 1.0, 9, op="one", integrator="Riemann")
    many = magnus.compute_sc(4, f, 0.0, 1.0, 9, op="many", integrator="Riemann")
    total = magnus.compute_sc(4, f, 0.0, 1.0, 9, op="sum", integrator="Riemann")

    assert one.shape == (3,)
    assert many.shape == (4, 3)
    assert total.shape == (3,)
    np.testing.assert_allclose(one, many[3], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(total, np.sum(many, axis=0), rtol=1e-11, atol=1e-12)


def test_spacecurve_vectorized_and_pointwise_sampling_match():
    t0 = 0.0
    tf = 2.0
    samples = 9
    n = 3
    seen = []

    def pointwise(t):
        assert isinstance(t, float)
        seen.append(t)
        return spacecurve_value(t)

    vectorized = magnus.compute_sc(
        n,
        spacecurve_values,
        t0,
        tf,
        samples,
        op="sum",
        integrator="Trapezoid",
    )
    point = magnus.compute_sc(
        n,
        pointwise,
        t0,
        tf,
        samples,
        op="sum",
        vectorized=False,
        integrator="Trapezoid",
    )

    np.testing.assert_array_equal(seen, np.linspace(t0, tf, samples))
    np.testing.assert_allclose(point, vectorized, rtol=1e-11, atol=1e-12)


def test_compute_can_return_vjp_data():
    output, vjp_data = magnus.compute(
        4,
        matrix_values,
        0.0,
        1.0,
        9,
        op="sum",
        record_vjp=True,
    )

    assert output.shape == (2, 2)
    assert vjp_data.shape == (3, 3, 9, 2, 2)


def test_compute_sc_can_return_vjp_data():
    output, vjp_data = magnus.compute_sc(
        4,
        spacecurve_values,
        0.0,
        1.0,
        9,
        op="sum",
        record_vjp=True,
    )

    assert output.shape == (3,)
    assert vjp_data.shape == (3, 3, 9, 2, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_compute_dtype_controls_sample_dtype(dtype):
    def f(t):
        data = matrix_values(t, dtype=np.float64)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            data = data + 0.25j * data
        return data

    actual = magnus.compute(
        1,
        f,
        0.0,
        1.0,
        9,
        op="one",
        dtype=dtype,
        matrix_backend="Manual",
        integrator="Riemann",
    )

    assert actual.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_compute_sc_dtype_controls_sample_dtype(dtype):
    def f(t):
        data = spacecurve_values(t, dtype=np.float64)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            data = data + 0.25j * data
        return data

    actual = magnus.compute_sc(
        1,
        f,
        0.0,
        1.0,
        9,
        op="one",
        dtype=dtype,
        integrator="Riemann",
    )

    assert actual.dtype == np.dtype(dtype)


def test_dispatch_kwargs_are_forwarded():
    def f(t):
        return np.zeros((np.asarray(t).size, 3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="invalid dispatch option"):
        magnus.compute(1, f, 0.0, 1.0, 9, op="one", matrix_backend="Fixed2")

    with pytest.raises(ValueError, match="dispatch name"):
        magnus.compute(1, f, 0.0, 1.0, 9, op="one", matrix_backend="Missing")

    with pytest.raises(ValueError, match="dispatch name"):
        magnus.compute(1, f, 0.0, 1.0, 9, op="one", integrator="Missing")

    with pytest.raises(ValueError, match="deduce kernel op"):
        magnus.compute(1, f, 0.0, 1.0, 9, op="missing")


def test_sample_count_is_validated_by_core_after_sampling():
    called = False

    def f(t):
        nonlocal called
        called = True
        return matrix_values(t)

    with pytest.raises(ValueError, match="at least two samples"):
        magnus.compute(1, f, 0.0, 1.0, 1)

    assert called


def test_spacecurve_sample_count_is_validated_by_core_after_sampling():
    called = False

    def f(t):
        nonlocal called
        called = True
        return spacecurve_values(t)

    with pytest.raises(ValueError, match="at least two samples"):
        magnus.compute_sc(1, f, 0.0, 1.0, 1)

    assert called


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

    with pytest.raises(ValueError):
        magnus.compute(1, f, 0.0, 1.0, 9)


@pytest.mark.parametrize(
    "bad_result",
    [
        np.zeros((9, 2), dtype=np.float64),
        np.zeros((8, 3), dtype=np.float64),
        np.zeros((9, 3, 1), dtype=np.float64),
    ],
)
def test_spacecurve_callable_output_shape_is_validated(bad_result):
    def f(t):
        return bad_result

    with pytest.raises(ValueError):
        magnus.compute_sc(1, f, 0.0, 1.0, 9)


def test_core_inputs_are_validated_through_public_compute():
    def bad_dtype(t):
        return matrix_values(t).astype(np.int64)

    with pytest.raises(ValueError, match="n must be at least 1"):
        magnus.compute(0, matrix_values, 0.0, 1.0, 9)

    with pytest.raises(TypeError, match="supports dtypes"):
        magnus.compute(1, bad_dtype, 0.0, 1.0, 9)


def test_spacecurve_core_inputs_are_validated_through_public_compute():
    def bad_dtype(t):
        return spacecurve_values(t).astype(np.int64)

    def bad_width(t):
        return np.zeros((np.asarray(t).size, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="n must be at least 1"):
        magnus.compute_sc(0, spacecurve_values, 0.0, 1.0, 9)

    with pytest.raises(TypeError, match="supports dtypes"):
        magnus.compute_sc(1, bad_dtype, 0.0, 1.0, 9)

    with pytest.raises(ValueError, match="shape"):
        magnus.compute_sc(1, bad_width, 0.0, 1.0, 9)
