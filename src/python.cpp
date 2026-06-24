#include "backendcomposer.hpp"
#include "gausslegendre.hpp"
#include "integratebackends.hpp"
#include "matrixbackends.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" std::byte _binary_gl_nodes_bin_start[];

namespace Magnus::detail {

template <class NumT>
using CArray = py::array_t<NumT, py::array::c_style>;

template <class NumT>
using CArrayCoercible = py::array_t<NumT, py::array::c_style | py::array::forcecast>;

size_t max_order() {
    return GLTable::get()->max_order();
}

void validate_input_shape(size_t n, const py::buffer_info& info) {
    if (n == 0) {
        throw py::value_error("n must be at least 1");
    }

    if (info.ndim != 3) {
        throw py::value_error("data must have shape (samples, dim, dim)");
    }

    if (info.shape[0] < 2) {
        throw py::value_error("data must contain at least two samples");
    }

    if (info.shape[1] <= 0 || info.shape[2] != info.shape[1]) {
        throw py::value_error("data must contain square matrices");
    }
}

std::vector<py::ssize_t> output_shape(
    Dispatch::KernelOp op,
    size_t n,
    py::ssize_t dim
) {
    if (op == Dispatch::KernelOp::MANY) {
        return {static_cast<py::ssize_t>(n), dim, dim};
    }

    return {dim, dim};
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
    if (!typed) {
        throw py::type_error("could not convert data to the selected numeric dtype");
    }

    py::buffer_info in_info = typed.request();
    validate_input_shape(n, in_info);

    const size_t samples = static_cast<size_t>(in_info.shape[0]);
    const size_t dim = static_cast<size_t>(in_info.shape[1]);

    CArray<NumT> out(output_shape(op, n, in_info.shape[1]));
    py::buffer_info out_info = out.request();

    Params params{
        Params::num_data<NumT>{
            static_cast<NumT*>(in_info.ptr),
            static_cast<NumT*>(out_info.ptr)
        },
        n,
        dim,
        samples,
        t0,
        tf
    };

    size_t num_idx = NumBackends::resolve(type_name_v<NumT>);
    size_t mat_idx = MatrixBackends::resolve(matrix_backend);
    size_t int_idx = IntegratorBackends::resolve(integrator);

    std::unique_ptr<KernelPlan> plan = make_plan(params, num_idx, mat_idx, int_idx);

    {
        py::gil_scoped_release release;
        plan->run(op);
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

    if (dtype.is(py::dtype::of<f32>())) {
        return run_typed<f32>(n, data, t0, tf, op, matrix_backend, integrator);
    }

    if (dtype.is(py::dtype::of<f64>())) {
        return run_typed<f64>(n, data, t0, tf, op, matrix_backend, integrator);
    }

    if (dtype.is(py::dtype::of<c32>())) {
        return run_typed<c32>(n, data, t0, tf, op, matrix_backend, integrator);
    }

    if (dtype.is(py::dtype::of<c64>())) {
        return run_typed<c64>(n, data, t0, tf, op, matrix_backend, integrator);
    }

    throw py::type_error(
        "magnus only supports dtypes float32, float64, complex64, and complex128"
    );
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

}

PYBIND11_MODULE(_core, m) {
    Magnus::GLTable::update(
        std::make_shared<Magnus::StaticTable>(_binary_gl_nodes_bin_start)
    );

    m.doc() = "Python bindings for the magnus C++ library";

    m.def(
        "max_order",
        &Magnus::detail::max_order,
        "Return the max available Gauss-Legendre order. Max magnus order is then any n satisfying (n + 3) / 2 >= max_order()"
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
}
