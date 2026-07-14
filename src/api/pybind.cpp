#include "composer/backends.hpp"
#include "util/gausslegendre.hpp"

#include <concepts>
#include <limits>
#include <memory>
#include <optional>
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

template <class Names>
py::tuple names_tuple(const Names& names) {
    py::tuple out(names.size());
    for (size_t i = 0; i < names.size(); ++i) out[i] = py::str(names[i]);
    return out;
}

template <class BackendList>
py::tuple dispatch_names_tuple() {
    constexpr size_t extra = BackendList::AutoEnabled ? 1 : 0;
    py::tuple out(BackendList::dispatch_names.size() + extra);

    size_t i = 0;
    for (; i < BackendList::dispatch_names.size(); ++i) out[i] = py::str(BackendList::dispatch_names[i]);

    if constexpr (BackendList::AutoEnabled) out[i] = py::str("Auto");

    return out;
}

py::tuple numeric_backends() { return dispatch_names_tuple<NumBackends>(); }
py::tuple matrix_backends() { return dispatch_names_tuple<MatrixBackends>(); }
py::tuple integrators() { return dispatch_names_tuple<IntegratorBackends>(); }
py::tuple ops() { return names_tuple(Dispatch::kernel_op_names); }

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

std::vector<py::ssize_t> py_shape(const std::vector<size_t>& shape) {
    std::vector<py::ssize_t> out;
    out.reserve(shape.size());

    for (size_t value : shape) {
        if (value > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max())) {
            throw py::value_error("output shape is too large");
        }

        out.push_back(static_cast<py::ssize_t>(value));
    }

    return out;
}

py::array vjp_array(py::dtype dtype, size_t n, size_t samples, size_t dim) {
    return py::array(
        dtype,
        py_shape({
            gl_max_n(n),
            total_orders(n),
            samples,
            dim,
            dim,
        })
    );
}

template <Numeric NumT>
py::object run_typed(
    size_t n,
    py::array data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view matrix_backend,
    std::string_view integrator,
    bool record_vjp
) {
    CArrayCoercible<NumT> typed = CArrayCoercible<NumT>::ensure(data);
    if (!typed) throw py::type_error("could not convert data to the selected numeric dtype");

    py::buffer_info in_info = typed.request();
    MatrixInputShape input_shape = matrix_input_shape(n, in_info.shape);

    CArray<NumT> out(py_shape(matrix_output_shape(op, n, input_shape.dim)));
    py::buffer_info out_info = out.request();
    NumT* in = static_cast<NumT*>(in_info.ptr);
    NumT* out_data = static_cast<NumT*>(out_info.ptr);
    py::array vjp = record_vjp ? vjp_array(data.dtype(), n, input_shape.samples, input_shape.dim) : py::array();
    NumT* vjp_data = record_vjp ? static_cast<NumT*>(vjp.request().ptr) : nullptr;

    {
        py::gil_scoped_release release;
        run_raw_impl(
            n,
            in,
            out_data,
            vjp_data,
            input_shape.samples,
            input_shape.dim,
            t0,
            tf,
            Dispatch::op_from_str(op),
            MatrixBackends::resolve(matrix_backend),
            IntegratorBackends::resolve(integrator),
            record_vjp
        );
    }

    if (record_vjp) return py::make_tuple(out, vjp);
    return out;
}

template <Numeric NumT>
py::object run_spacecurve_typed(
    size_t n,
    py::array data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view integrator,
    bool record_vjp
) {
    CArrayCoercible<NumT> typed = CArrayCoercible<NumT>::ensure(data);
    if (!typed) throw py::type_error("could not convert data to the selected numeric dtype");

    py::buffer_info in_info = typed.request();
    SpaceCurveInputShape input_shape = spacecurve_input_shape(n, in_info.shape);
    NumT* in = static_cast<NumT*>(in_info.ptr);

    CArray<NumT> out(py_shape(spacecurve_output_shape(op, n)));
    py::buffer_info out_info = out.request();
    NumT* out_data = static_cast<NumT*>(out_info.ptr);
    py::array vjp = record_vjp ? vjp_array(data.dtype(), n, input_shape.samples, 2) : py::array();
    NumT* vjp_data = record_vjp ? static_cast<NumT*>(vjp.request().ptr) : nullptr;

    {
        py::gil_scoped_release release;
        run_spacecurve_raw_impl(
            n,
            in,
            out_data,
            vjp_data,
            input_shape.samples,
            t0,
            tf,
            Dispatch::op_from_str(op),
            IntegratorBackends::resolve(integrator),
            record_vjp
        );
    }

    if (record_vjp) return py::make_tuple(out, vjp);
    return out;
}

template <Numeric NumT>
py::array run_vjp_typed(
    size_t n,
    py::array data,
    py::array cotangent,
    py::object vjp_data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view matrix_backend,
    std::string_view integrator
) {
    CArrayCoercible<NumT> typed = CArrayCoercible<NumT>::ensure(data);
    CArrayCoercible<NumT> typed_cotangent = CArrayCoercible<NumT>::ensure(cotangent);
    if (!typed) throw py::type_error("could not convert data to the selected numeric dtype");
    if (!typed_cotangent) throw py::type_error("could not convert cotangent to the selected numeric dtype");

    py::buffer_info in_info = typed.request();
    py::buffer_info cotangent_info = typed_cotangent.request();
    MatrixInputShape input_shape = matrix_input_shape(n, in_info.shape);
    validate_matrix_output_shape(op, n, input_shape.dim, cotangent_info.shape);

    std::optional<CArrayCoercible<NumT>> typed_vjp_data;
    NumT* carry = nullptr;
    if (!vjp_data.is_none()) {
        typed_vjp_data.emplace(CArrayCoercible<NumT>::ensure(vjp_data));
        if (!*typed_vjp_data) throw py::type_error("could not convert VJP data to the selected numeric dtype");

        py::buffer_info vjp_info = typed_vjp_data->request();
        validate_vjp_data_shape(n, input_shape.samples, input_shape.dim, vjp_info.shape);
        carry = static_cast<NumT*>(vjp_info.ptr);
    }

    CArray<NumT> out({
        static_cast<py::ssize_t>(input_shape.samples),
        static_cast<py::ssize_t>(input_shape.dim),
        static_cast<py::ssize_t>(input_shape.dim),
    });
    py::buffer_info out_info = out.request();

    {
        py::gil_scoped_release release;
        run_vjp_raw_impl(
            n,
            static_cast<NumT*>(in_info.ptr),
            static_cast<NumT*>(cotangent_info.ptr),
            static_cast<NumT*>(out_info.ptr),
            carry,
            input_shape.samples,
            input_shape.dim,
            t0,
            tf,
            Dispatch::op_from_str(op),
            MatrixBackends::resolve(matrix_backend),
            IntegratorBackends::resolve(integrator)
        );
    }

    return out;
}

template <Numeric NumT>
py::array run_spacecurve_vjp_typed(
    size_t n,
    py::array data,
    py::array cotangent,
    py::object vjp_data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view integrator
) {
    CArrayCoercible<NumT> typed = CArrayCoercible<NumT>::ensure(data);
    CArrayCoercible<NumT> typed_cotangent = CArrayCoercible<NumT>::ensure(cotangent);
    if (!typed) throw py::type_error("could not convert data to the selected numeric dtype");
    if (!typed_cotangent) throw py::type_error("could not convert cotangent to the selected numeric dtype");

    py::buffer_info in_info = typed.request();
    py::buffer_info cotangent_info = typed_cotangent.request();
    SpaceCurveInputShape input_shape = spacecurve_input_shape(n, in_info.shape);
    validate_spacecurve_output_shape(op, n, cotangent_info.shape);

    std::optional<CArrayCoercible<NumT>> typed_vjp_data;
    NumT* carry = nullptr;
    if (!vjp_data.is_none()) {
        typed_vjp_data.emplace(CArrayCoercible<NumT>::ensure(vjp_data));
        if (!*typed_vjp_data) throw py::type_error("could not convert VJP data to the selected numeric dtype");

        py::buffer_info vjp_info = typed_vjp_data->request();
        validate_vjp_data_shape(n, input_shape.samples, 2, vjp_info.shape);
        carry = static_cast<NumT*>(vjp_info.ptr);
    }

    CArray<NumT> out({
        static_cast<py::ssize_t>(input_shape.samples),
        py::ssize_t{3},
    });
    py::buffer_info out_info = out.request();

    {
        py::gil_scoped_release release;
        run_spacecurve_vjp_raw_impl(
            n,
            static_cast<NumT*>(in_info.ptr),
            static_cast<NumT*>(cotangent_info.ptr),
            static_cast<NumT*>(out_info.ptr),
            carry,
            input_shape.samples,
            t0,
            tf,
            Dispatch::op_from_str(op),
            IntegratorBackends::resolve(integrator)
        );
    }

    return out;
}

py::object compute(
    size_t n,
    py::array data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view matrix_backend,
    std::string_view integrator,
    bool record_vjp
) {
    py::dtype dtype = data.dtype();

    if (dtype.is(py::dtype::of<f32>())) return run_typed<f32>(n, data, t0, tf, op, matrix_backend, integrator, record_vjp);
    if (dtype.is(py::dtype::of<f64>())) return run_typed<f64>(n, data, t0, tf, op, matrix_backend, integrator, record_vjp);
    if (dtype.is(py::dtype::of<c32>())) return run_typed<c32>(n, data, t0, tf, op, matrix_backend, integrator, record_vjp);
    if (dtype.is(py::dtype::of<c64>())) return run_typed<c64>(n, data, t0, tf, op, matrix_backend, integrator, record_vjp);
    throw py::type_error("magnus only supports dtypes float32, float64, complex64, and complex128");
}

py::object compute_sc(
    size_t n,
    py::array data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view integrator,
    bool record_vjp
) {
    py::dtype dtype = data.dtype();

    if (dtype.is(py::dtype::of<f32>())) return run_spacecurve_typed<f32>(n, data, t0, tf, op, integrator, record_vjp);
    if (dtype.is(py::dtype::of<f64>())) return run_spacecurve_typed<f64>(n, data, t0, tf, op, integrator, record_vjp);
    if (dtype.is(py::dtype::of<c32>())) return run_spacecurve_typed<c32>(n, data, t0, tf, op, integrator, record_vjp);
    if (dtype.is(py::dtype::of<c64>())) return run_spacecurve_typed<c64>(n, data, t0, tf, op, integrator, record_vjp);
    throw py::type_error("magnus only supports dtypes float32, float64, complex64, and complex128");
}

py::array compute_vjp(
    size_t n,
    py::array data,
    py::array cotangent,
    py::object vjp_data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view matrix_backend,
    std::string_view integrator
) {
    py::dtype dtype = data.dtype();

    if (dtype.is(py::dtype::of<f32>())) return run_vjp_typed<f32>(n, data, cotangent, vjp_data, t0, tf, op, matrix_backend, integrator);
    if (dtype.is(py::dtype::of<f64>())) return run_vjp_typed<f64>(n, data, cotangent, vjp_data, t0, tf, op, matrix_backend, integrator);
    if (dtype.is(py::dtype::of<c32>())) return run_vjp_typed<c32>(n, data, cotangent, vjp_data, t0, tf, op, matrix_backend, integrator);
    if (dtype.is(py::dtype::of<c64>())) return run_vjp_typed<c64>(n, data, cotangent, vjp_data, t0, tf, op, matrix_backend, integrator);
    throw py::type_error("magnus only supports dtypes float32, float64, complex64, and complex128");
}

py::array compute_sc_vjp(
    size_t n,
    py::array data,
    py::array cotangent,
    py::object vjp_data,
    double t0,
    double tf,
    std::string_view op,
    std::string_view integrator
) {
    py::dtype dtype = data.dtype();

    if (dtype.is(py::dtype::of<f32>())) return run_spacecurve_vjp_typed<f32>(n, data, cotangent, vjp_data, t0, tf, op, integrator);
    if (dtype.is(py::dtype::of<f64>())) return run_spacecurve_vjp_typed<f64>(n, data, cotangent, vjp_data, t0, tf, op, integrator);
    if (dtype.is(py::dtype::of<c32>())) return run_spacecurve_vjp_typed<c32>(n, data, cotangent, vjp_data, t0, tf, op, integrator);
    if (dtype.is(py::dtype::of<c64>())) return run_spacecurve_vjp_typed<c64>(n, data, cotangent, vjp_data, t0, tf, op, integrator);
    throw py::type_error("magnus only supports dtypes float32, float64, complex64, and complex128");
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
        "numeric_backends",
        &Magnus::detail::numeric_backends,
        "Return the available numeric backend dispatch names."
    );

    m.def(
        "matrix_backends",
        &Magnus::detail::matrix_backends,
        "Return the available matrix backend dispatch names."
    );

    m.def(
        "integrators",
        &Magnus::detail::integrators,
        "Return the available integrator dispatch names."
    );

    m.def(
        "ops",
        &Magnus::detail::ops,
        "Return the available kernel operation dispatch names."
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
        "compute",
        &Magnus::detail::compute,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("op") = "sum",
        py::arg("matrix_backend") = "Auto",
        py::arg("integrator") = "Auto",
        py::arg("record_vjp") = false,
        "Compute a Magnus operation from sampled matrix data."
    );

    m.def(
        "compute_sc",
        &Magnus::detail::compute_sc,
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("op") = "sum",
        py::arg("integrator") = "Auto",
        py::arg("record_vjp") = false,
        "Compute a SpaceCurve Magnus operation from sampled 3-vector data."
    );

    m.def(
        "compute_vjp",
        &Magnus::detail::compute_vjp,
        py::arg("n"),
        py::arg("data"),
        py::arg("cotangent"),
        py::arg("vjp_data") = py::none(),
        py::arg("t0") = 0.0,
        py::arg("tf") = 1.0,
        py::arg("op") = "sum",
        py::arg("matrix_backend") = "Auto",
        py::arg("integrator") = "Auto",
        "Compute the VJP of a Magnus operation with respect to sampled matrix data."
    );

    m.def(
        "compute_sc_vjp",
        &Magnus::detail::compute_sc_vjp,
        py::arg("n"),
        py::arg("data"),
        py::arg("cotangent"),
        py::arg("vjp_data") = py::none(),
        py::arg("t0") = 0.0,
        py::arg("tf") = 1.0,
        py::arg("op") = "sum",
        py::arg("integrator") = "Auto",
        "Compute the VJP of a SpaceCurve Magnus operation with respect to sampled vector data."
    );

#ifdef MAGNUS_ENABLE_JAX_FFI
    Magnus::JaxFfi::bind(m);
#endif
}
