#include "backendcomposer.hpp"
#include "gausslegendre.hpp"

#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef MAGNUS_ENABLE_JAX_FFI
namespace Magnus::JaxFfi {
void bind(py::module_& m);
}
#endif

namespace Magnus::detail {

template <class NumT>
using CArray = py::array_t<NumT, py::array::c_style>;

template <class NumT>
using CArrayCoercible = py::array_t<NumT, py::array::c_style | py::array::forcecast>;

size_t max_order() {
    return GLTable::get()->max_order();
}

size_t gl_entry_count(size_t max_order) {
    if (max_order == 0) throw py::value_error("max_order must be at least 1");

    size_t a = max_order;
    size_t b = max_order + 1;

    if (b == 0) throw py::value_error("max_order is too large");

    if (a % 2 == 0) a /= 2;
    else b /= 2;

    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) throw py::value_error("max_order is too large");

    return a * b;
}

std::vector<size_t> gl_offsets(size_t max_order) {
    std::vector<size_t> offsets(max_order + 1);
    size_t offset = 0;
    offsets[0] = 0;

    for (size_t order = 1; order <= max_order; ++order) {
        offsets[order] = offset;
        offset += order;
    }

    return offsets;
}

std::vector<double> copy_gl_array(py::array data, size_t expected_size, std::string_view name) {
    CArrayCoercible<double> typed = CArrayCoercible<double>::ensure(data);
    if (!typed) throw py::type_error("GL data must be convertible to float64");

    py::buffer_info info = typed.request();
    if (info.ndim != 1 || static_cast<size_t>(info.shape[0]) != expected_size)
        throw py::value_error("GL " + std::string(name) + " values must have shape ("+ std::to_string(expected_size) + ",)");

    const double* ptr = static_cast<const double*>(info.ptr);
    return {ptr, ptr + expected_size};
}

void replace_gl_table(size_t max_order, py::array weights, py::array nodes) {
    size_t expected_size = gl_entry_count(max_order);

    std::vector<double> weight_values = copy_gl_array(weights, expected_size, "weights");
    std::vector<double> node_values = copy_gl_array(nodes, expected_size, "nodes");

    GLTable::update(
        std::make_shared<OwningTable>(
            max_order,
            gl_offsets(max_order),
            std::move(weight_values),
            std::move(node_values)
        )
    );
}

void validate_input_shape(size_t n, const py::buffer_info& info) {
    if (n == 0) throw py::value_error("n must be at least 1");
    if (info.ndim != 3) throw py::value_error("data must have shape (samples, dim, dim)");
    if (info.shape[0] < 2) throw py::value_error("data must contain at least two samples");
    if (info.shape[1] <= 0 || info.shape[2] != info.shape[1]) throw py::value_error("data must contain square matrices");
}

void validate_spacecurve_input_shape(size_t n, const py::buffer_info& info) {
    if (n == 0) throw py::value_error("n must be at least 1");
    if (info.ndim != 2) throw py::value_error("data must have shape (samples, 3)");
    if (info.shape[0] < 2)  throw py::value_error("data must contain at least two samples");
    if (info.shape[1] != 3) throw py::value_error("data must have shape (samples, 3)");
}

std::vector<py::ssize_t> output_shape(Dispatch::KernelOp op, size_t n, py::ssize_t dim) {
    if (op == Dispatch::KernelOp::MANY) return {static_cast<py::ssize_t>(n), dim, dim};

    return {dim, dim};
}

std::vector<py::ssize_t> spacecurve_output_shape(Dispatch::KernelOp op, size_t n) {
    if (op == Dispatch::KernelOp::MANY) return {static_cast<py::ssize_t>(n), 3};

    return {3};
}

template <Numeric NumT>
py::array run_typed(
    size_t n,
    py::array data,
    double t0,
    double tf,
    Dispatch::KernelOp op,
    std::string_view matrix_backend,
    std::string_view integrator
) {
    CArrayCoercible<NumT> typed = CArrayCoercible<NumT>::ensure(data);
    if (!typed) throw py::type_error("could not convert data to the selected numeric dtype");

    py::buffer_info in_info = typed.request();
    validate_input_shape(n, in_info);

    const size_t samples = static_cast<size_t>(in_info.shape[0]);
    const size_t dim = static_cast<size_t>(in_info.shape[1]);

    CArray<NumT> out(output_shape(op, n, in_info.shape[1]));
    py::buffer_info out_info = out.request();
    const NumT* in = static_cast<const NumT*>(in_info.ptr);
    NumT* out_data = static_cast<NumT*>(out_info.ptr);

    {
        py::gil_scoped_release release;
        run_raw(
            n,
            in,
            out_data,
            samples,
            dim,
            t0,
            tf,
            op,
            matrix_backend,
            integrator
        );
    }

    return out;
}

template <Numeric NumT>
py::array run_spacecurve_typed(
    size_t n,
    py::array data,
    double t0,
    double tf,
    Dispatch::KernelOp op,
    std::string_view integrator
) {
    CArrayCoercible<NumT> typed = CArrayCoercible<NumT>::ensure(data);
    if (!typed) throw py::type_error("could not convert data to the selected numeric dtype");

    py::buffer_info in_info = typed.request();
    validate_spacecurve_input_shape(n, in_info);

    const size_t samples = static_cast<size_t>(in_info.shape[0]);
    const NumT* in = static_cast<const NumT*>(in_info.ptr);

    CArray<NumT> out(spacecurve_output_shape(op, n));
    py::buffer_info out_info = out.request();
    NumT* out_data = static_cast<NumT*>(out_info.ptr);

    {
        py::gil_scoped_release release;
        run_spacecurve_raw(
            n,
            in,
            out_data,
            samples,
            t0,
            tf,
            op,
            integrator
        );
    }

    return out;
}

py::array run(
    size_t n,
    py::array data,
    double t0,
    double tf,
    Dispatch::KernelOp op,
    const std::string& matrix_backend,
    const std::string& integrator
) {
    py::dtype dtype = data.dtype();

    if (dtype.is(py::dtype::of<f32>())) return run_typed<f32>(n, data, t0, tf, op, matrix_backend, integrator);
    if (dtype.is(py::dtype::of<f64>())) return run_typed<f64>(n, data, t0, tf, op, matrix_backend, integrator);
    if (dtype.is(py::dtype::of<c32>())) return run_typed<c32>(n, data, t0, tf, op, matrix_backend, integrator);
    if (dtype.is(py::dtype::of<c64>())) return run_typed<c64>(n, data, t0, tf, op, matrix_backend, integrator);
    throw py::type_error("magnus only supports dtypes float32, float64, complex64, and complex128");
}

py::array run_spacecurve(
    size_t n,
    py::array data,
    double t0,
    double tf,
    Dispatch::KernelOp op,
    const std::string& integrator
) {
    py::dtype dtype = data.dtype();

    if (dtype.is(py::dtype::of<f32>())) return run_spacecurve_typed<f32>(n, data, t0, tf, op, integrator);
    if (dtype.is(py::dtype::of<f64>())) return run_spacecurve_typed<f64>(n, data, t0, tf, op, integrator);
    if (dtype.is(py::dtype::of<c32>())) return run_spacecurve_typed<c32>(n, data, t0, tf, op, integrator);
    if (dtype.is(py::dtype::of<c64>())) return run_spacecurve_typed<c64>(n, data, t0, tf, op, integrator);
    throw py::type_error("magnus only supports dtypes float32, float64, complex64, and complex128");
}

py::array magnus_one(
    size_t n,
    py::array data,
    double t0,
    double tf,
    const std::string& matrix_backend,
    const std::string& integrator
) {
    return run(n, data, t0, tf, Dispatch::KernelOp::ONE, matrix_backend, integrator);
}

py::array magnus_many(
    size_t n,
    py::array data,
    double t0,
    double tf,
    const std::string& matrix_backend,
    const std::string& integrator
) {
    return run(n, data, t0, tf, Dispatch::KernelOp::MANY, matrix_backend, integrator);
}

py::array magnus_sum(
    size_t n,
    py::array data,
    double t0,
    double tf,
    const std::string& matrix_backend,
    const std::string& integrator
) {
    return run(n, data, t0, tf, Dispatch::KernelOp::SUM, matrix_backend, integrator);
}

py::array magnus_spacecurve_one(
    size_t n,
    py::array data,
    double t0,
    double tf,
    const std::string& integrator
) {
    return run_spacecurve(n, data, t0, tf, Dispatch::KernelOp::ONE, integrator);
}

py::array magnus_spacecurve_many(
    size_t n,
    py::array data,
    double t0,
    double tf,
    const std::string& integrator
) {
    return run_spacecurve(n, data, t0, tf, Dispatch::KernelOp::MANY, integrator);
}

py::array magnus_spacecurve_sum(
    size_t n,
    py::array data,
    double t0,
    double tf,
    const std::string& integrator
) {
    return run_spacecurve(n, data, t0, tf, Dispatch::KernelOp::SUM, integrator);
}

}

PYBIND11_MODULE(_core, m) {
    Magnus::initialize_default_gl_table();

    m.doc() = "Python bindings for the magnus C++ library";

    m.def(
        "max_order",
        &Magnus::detail::max_order,
        "Return the max available Gauss-Legendre order. Max magnus order is then any n satisfying (n + 3) / 2 <= max_order()."
    );

    m.def(
        "_replace_gl_table",
        &Magnus::detail::replace_gl_table,
        py::arg("max_order"),
        py::arg("weights"),
        py::arg("nodes"),
        "Replace the process-global Gauss-Legendre table from flattened float64 weights and nodes."
    );

    m.def(
        "one",
        &Magnus::detail::magnus_one,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("matrix_backend") = "Auto",
        py::arg("integrator") = "Auto",
        "Compute exactly the nth term of the Magnus expansion."
    );

    m.def(
        "many",
        &Magnus::detail::magnus_many,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("matrix_backend") = "Auto",
        py::arg("integrator") = "Auto",
        "Compute all Magnus expansion terms from 1 to n."
    );

    m.def(
        "sum",
        &Magnus::detail::magnus_sum,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("matrix_backend") = "Auto",
        py::arg("integrator") = "Auto",
        "Compute the sum of Magnus expansion terms from 1 to n."
    );

    m.def(
        "one_sc",
        &Magnus::detail::magnus_spacecurve_one,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("integrator") = "Auto",
        "Compute exactly the nth SpaceCurve Magnus expansion term from 3-vector samples."
    );

    m.def(
        "many_sc",
        &Magnus::detail::magnus_spacecurve_many,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("integrator") = "Auto",
        "Compute SpaceCurve Magnus expansion terms from 1 to n from 3-vector samples."
    );

    m.def(
        "sum_sc",
        &Magnus::detail::magnus_spacecurve_sum,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("integrator") = "Auto",
        "Compute the sum of SpaceCurve Magnus expansion terms from 3-vector samples."
    );

#ifdef MAGNUS_ENABLE_JAX_FFI
    Magnus::JaxFfi::bind(m);
#endif
}
