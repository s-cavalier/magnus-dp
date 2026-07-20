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


def gradient_values(shape, dtype, real_bounds, imaginary_bounds):
    dtype = np.dtype(dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    real = np.linspace(*real_bounds, np.prod(shape), dtype=real_dtype).reshape(shape)
    if np.issubdtype(dtype, np.complexfloating):
        imaginary = np.linspace(*imaginary_bounds, np.prod(shape), dtype=real_dtype).reshape(shape)
        real = real + 1j * imaginary
    return np.asarray(real, dtype=dtype)


def dense_matrix_values(samples, dim, dtype=np.float64):
    dtype = np.dtype(dtype)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    t = np.linspace(0.0, 1.0, samples, dtype=real_dtype)[:, None, None]
    row = np.arange(1, dim + 1, dtype=real_dtype)[None, :, None]
    column = np.arange(1, dim + 1, dtype=real_dtype)[None, None, :]
    data = np.sin((row + 0.3 * column) * t) + 0.1 * row - 0.07 * column
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * (np.cos((0.2 * row + column) * t) - 0.15 * t)
    return np.asarray(data, dtype=dtype)


def assert_matrix_vjp_directional(
    data,
    *,
    n=4,
    op="sum",
    matrix_backend="Fixed2",
    integrator="Boole",
    epsilon=1e-6,
    rtol=1e-7,
    atol=1e-9,
):
    samples = data.shape[0]
    direction = gradient_values(data.shape, data.dtype, (-0.3, 0.7), (0.5, -0.2))

    def f(_):
        return data

    kwargs = {
        "op": op,
        "dtype": data.dtype,
        "matrix_backend": matrix_backend,
        "integrator": integrator,
    }
    out, vjp_data = magnus.compute(
        n,
        f,
        0.0,
        1.0,
        samples,
        record_vjp=True,
        **kwargs,
    )
    cotangent = gradient_values(out.shape, data.dtype, (0.2, 1.1), (-0.4, 0.3))
    actual = magnus.compute_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        vjp_data=vjp_data,
        **kwargs,
    )
    recomputed = magnus.compute_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        **kwargs,
    )

    perturbation = np.asarray(epsilon * direction, dtype=data.dtype)
    plus = magnus.compute(
        n,
        lambda _: data + perturbation,
        0.0,
        1.0,
        samples,
        **kwargs,
    )
    minus = magnus.compute(
        n,
        lambda _: data - perturbation,
        0.0,
        1.0,
        samples,
        **kwargs,
    )

    # Complex kernels expose the analytic, bilinear VJP rather than a Hermitian pairing.
    expected_directional = np.sum((plus - minus) * cotangent) / (2 * epsilon)
    actual_directional = np.sum(actual * direction)
    np.testing.assert_allclose(actual_directional, expected_directional, rtol=rtol, atol=atol)

    real_dtype = np.empty((), dtype=data.dtype).real.dtype
    carry_tolerance = max(1e-12, 20 * np.finfo(real_dtype).eps)
    np.testing.assert_allclose(
        recomputed,
        actual,
        rtol=carry_tolerance,
        atol=carry_tolerance,
    )


def assert_spacecurve_vjp_directional(
    data,
    *,
    n=4,
    op="sum",
    integrator="Boole",
    epsilon=1e-6,
    rtol=1e-7,
    atol=1e-9,
):
    samples = data.shape[0]
    direction = gradient_values(data.shape, data.dtype, (-0.4, 0.6), (0.3, -0.5))

    def f(_):
        return data

    kwargs = {
        "op": op,
        "dtype": data.dtype,
        "integrator": integrator,
    }
    out, vjp_data = magnus.compute_sc(
        n,
        f,
        0.0,
        1.0,
        samples,
        record_vjp=True,
        **kwargs,
    )
    cotangent = gradient_values(out.shape, data.dtype, (0.2, 1.1), (-0.3, 0.4))
    actual = magnus.compute_sc_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        vjp_data=vjp_data,
        **kwargs,
    )
    recomputed = magnus.compute_sc_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        **kwargs,
    )

    perturbation = np.asarray(epsilon * direction, dtype=data.dtype)
    plus = magnus.compute_sc(
        n,
        lambda _: data + perturbation,
        0.0,
        1.0,
        samples,
        **kwargs,
    )
    minus = magnus.compute_sc(
        n,
        lambda _: data - perturbation,
        0.0,
        1.0,
        samples,
        **kwargs,
    )

    # SpaceCurve follows the same analytic, bilinear convention as matrix kernels.
    expected_directional = np.sum((plus - minus) * cotangent) / (2 * epsilon)
    actual_directional = np.sum(actual * direction)
    np.testing.assert_allclose(actual_directional, expected_directional, rtol=rtol, atol=atol)

    real_dtype = np.empty((), dtype=data.dtype).real.dtype
    carry_tolerance = max(1e-12, 20 * np.finfo(real_dtype).eps)
    np.testing.assert_allclose(
        recomputed,
        actual,
        rtol=carry_tolerance,
        atol=carry_tolerance,
    )


def test_public_api_exports_only_compute_entrypoints():
    assert set(magnus.__all__) == {
        "max_order",
        "numeric_backends",
        "matrix_backends",
        "integrators",
        "gl_backends",
        "ops",
        "replace_gl_table",
        "compute",
        "compute_sc",
        "compute_vjp",
        "compute_sc_vjp",
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
    assert "serial" in magnus.gl_backends()
    assert "openmp" in magnus.gl_backends()
    assert "threadpool" in magnus.gl_backends()
    assert "Auto" in magnus.gl_backends()
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


def test_gl_backend_dispatch_and_auto_selection():
    small_samples = 33
    large_samples = 1025

    serial_small = magnus.compute(
        4,
        matrix_values,
        0.0,
        1.0,
        small_samples,
        integrator="Boole",
        gl_backend="serial",
    )
    auto_small = magnus.compute(
        4,
        matrix_values,
        0.0,
        1.0,
        small_samples,
        integrator="Boole",
        gl_backend="Auto",
    )
    np.testing.assert_array_equal(auto_small, serial_small)

    parallel_large, parallel_carry = magnus.compute(
        8,
        matrix_values,
        0.0,
        1.0,
        large_samples,
        integrator="Boole",
        gl_backend="openmp",
        record_vjp=True,
    )
    auto_large, auto_carry = magnus.compute(
        8,
        matrix_values,
        0.0,
        1.0,
        large_samples,
        integrator="Boole",
        gl_backend="Auto",
        record_vjp=True,
    )
    np.testing.assert_array_equal(auto_large, parallel_large)
    np.testing.assert_array_equal(auto_carry, parallel_carry)

    serial_sc = magnus.compute_sc(
        8,
        spacecurve_values,
        0.0,
        1.0,
        small_samples,
        integrator="Boole",
        gl_backend="serial",
    )
    parallel_sc = magnus.compute_sc(
        8,
        spacecurve_values,
        0.0,
        1.0,
        small_samples,
        integrator="Boole",
        gl_backend="openmp",
    )
    np.testing.assert_allclose(parallel_sc, serial_sc, rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize("op", ["one", "many", "sum"])
def test_compute_vjp_matches_directional_finite_difference(op):
    samples = 9
    n = 4
    data = matrix_values(np.linspace(0.0, 1.0, samples))
    direction = np.linspace(-0.3, 0.7, data.size).reshape(data.shape)

    def f(_):
        return data

    out, vjp_data = magnus.compute(
        n,
        f,
        0.0,
        1.0,
        samples,
        op=op,
        matrix_backend="Fixed2",
        integrator="Boole",
        record_vjp=True,
    )
    cotangent = np.linspace(0.2, 1.1, out.size).reshape(out.shape)
    actual = magnus.compute_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        op=op,
        matrix_backend="Fixed2",
        integrator="Boole",
        vjp_data=vjp_data,
    )
    recomputed = magnus.compute_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        op=op,
        matrix_backend="Fixed2",
        integrator="Boole",
    )

    epsilon = 1e-6
    data += epsilon * direction
    plus = magnus.compute(n, f, 0.0, 1.0, samples, op=op, matrix_backend="Fixed2", integrator="Boole")
    data -= 2 * epsilon * direction
    minus = magnus.compute(n, f, 0.0, 1.0, samples, op=op, matrix_backend="Fixed2", integrator="Boole")
    data += epsilon * direction

    expected_directional = np.sum((plus - minus) * cotangent) / (2 * epsilon)
    actual_directional = np.sum(actual * direction)
    np.testing.assert_allclose(actual_directional, expected_directional, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(recomputed, actual, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op", ["one", "many", "sum"])
def test_compute_sc_vjp_matches_directional_finite_difference(op):
    samples = 9
    n = 4
    data = spacecurve_values(np.linspace(0.0, 1.0, samples))
    direction = np.linspace(-0.4, 0.6, data.size).reshape(data.shape)

    def f(_):
        return data

    out, vjp_data = magnus.compute_sc(
        n,
        f,
        0.0,
        1.0,
        samples,
        op=op,
        integrator="Boole",
        record_vjp=True,
    )
    cotangent = np.linspace(0.2, 1.1, out.size).reshape(out.shape)
    actual = magnus.compute_sc_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        op=op,
        integrator="Boole",
        vjp_data=vjp_data,
    )
    recomputed = magnus.compute_sc_vjp(
        n,
        f,
        cotangent,
        0.0,
        1.0,
        samples,
        op=op,
        integrator="Boole",
    )

    epsilon = 1e-6
    data += epsilon * direction
    plus = magnus.compute_sc(n, f, 0.0, 1.0, samples, op=op, integrator="Boole")
    data -= 2 * epsilon * direction
    minus = magnus.compute_sc(n, f, 0.0, 1.0, samples, op=op, integrator="Boole")
    data += epsilon * direction

    expected_directional = np.sum((plus - minus) * cotangent) / (2 * epsilon)
    actual_directional = np.sum(actual * direction)
    np.testing.assert_allclose(actual_directional, expected_directional, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(recomputed, actual, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("integrator", ["Riemann", "Trapezoid", "Simpson", "Boole"])
def test_compute_vjp_matches_finite_difference_for_each_integrator(integrator):
    data = matrix_values(np.linspace(0.0, 1.0, 9))
    assert_matrix_vjp_directional(data, integrator=integrator)


@pytest.mark.parametrize("integrator", ["Riemann", "Trapezoid", "Simpson", "Boole"])
def test_compute_sc_vjp_matches_finite_difference_for_each_integrator(integrator):
    data = spacecurve_values(np.linspace(0.0, 1.0, 9))
    assert_spacecurve_vjp_directional(data, integrator=integrator)


@pytest.mark.parametrize("matrix_backend", ["Fixed2", "Manual", "Blas"])
def test_compute_vjp_matches_finite_difference_for_each_matrix_backend(matrix_backend):
    dim = 2 if matrix_backend == "Fixed2" else 3
    data = dense_matrix_values(9, dim)
    assert_matrix_vjp_directional(data, op="many", matrix_backend=matrix_backend)


@pytest.mark.parametrize(
    ("n", "samples"),
    [(1, 5), (2, 9), (3, 13), (6, 17)],
)
def test_compute_vjp_matches_finite_difference_across_orders_and_sample_counts(n, samples):
    data = matrix_values(np.linspace(0.0, 1.0, samples))
    assert_matrix_vjp_directional(data, n=n)


@pytest.mark.parametrize(
    ("n", "samples"),
    [(1, 5), (2, 9), (3, 13), (6, 17)],
)
def test_compute_sc_vjp_matches_finite_difference_across_orders_and_sample_counts(n, samples):
    data = spacecurve_values(np.linspace(0.0, 1.0, samples))
    assert_spacecurve_vjp_directional(data, n=n)


@pytest.mark.parametrize(
    ("dtype", "epsilon", "rtol", "atol"),
    [
        (np.float32, 2e-3, 3e-3, 3e-4),
        (np.complex64, 2e-3, 4e-3, 4e-4),
        (np.complex128, 1e-6, 2e-7, 2e-9),
    ],
)
def test_compute_vjp_matches_finite_difference_for_supported_gradient_dtypes(
    dtype,
    epsilon,
    rtol,
    atol,
):
    data = dense_matrix_values(9, 2, dtype=dtype)
    assert_matrix_vjp_directional(
        data,
        epsilon=epsilon,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("dtype", "epsilon", "rtol", "atol"),
    [
        (np.float32, 2e-3, 3e-3, 3e-4),
        (np.complex64, 2e-3, 4e-3, 4e-4),
        (np.complex128, 1e-6, 2e-7, 2e-9),
    ],
)
def test_compute_sc_vjp_matches_finite_difference_for_supported_gradient_dtypes(
    dtype,
    epsilon,
    rtol,
    atol,
):
    data = spacecurve_values(np.linspace(0.0, 1.0, 9), dtype=dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        imaginary = gradient_values(data.shape, dtype, (0.0, 0.0), (-0.2, 0.4))
        data = data + imaginary
    assert_spacecurve_vjp_directional(
        data,
        epsilon=epsilon,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("matrix_backend", ["Fixed2", "Manual", "Blas"])
def test_complex_compute_vjp_matches_finite_difference_for_each_matrix_backend(matrix_backend):
    dim = 2 if matrix_backend == "Fixed2" else 3
    data = dense_matrix_values(9, dim, dtype=np.complex128)
    assert_matrix_vjp_directional(
        data,
        op="many",
        matrix_backend=matrix_backend,
        rtol=2e-7,
        atol=2e-9,
    )


def test_compute_vjp_recomputation_accounts_for_nested_integrator_workspace():
    samples = 9
    dim = 16
    data = np.zeros((samples, dim, dim), dtype=np.float64)

    actual = magnus.compute_vjp(
        2,
        lambda _: data,
        np.ones((dim, dim), dtype=np.float64),
        0.0,
        1.0,
        samples,
        matrix_backend="Manual",
        integrator="Boole",
    )

    assert actual.shape == data.shape
    assert np.isfinite(actual).all()


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

    with pytest.raises(ValueError, match="dispatch name"):
        magnus.compute(1, f, 0.0, 1.0, 9, op="one", gl_backend="Missing")

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
