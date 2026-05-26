#include "integrate.hpp"
#include "matrix.hpp"
#include "magnus.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" std::byte _binary_gl_nodes_bin_start[];

namespace Magnus::detail {

template <class NumT>
using CArray = py::array_t<NumT, py::array::c_style>;

template <class NumT>
using CArray_Coercible = py::array_t<NumT, py::array::c_style | py::array::forcecast>;

size_t max_order() {
    return GLTable::get()->max_order();
}

template <class NumT>
CArray<NumT> magnus_one( size_t n, CArray_Coercible<NumT> data, double t0, double tf ) {
    auto info = data.request();
    
    if ( info.ndim != 3 ) throw py::value_error("data must have size (samples, dim, dim)");

    size_t n_samples = (size_t)(info.shape[0]);
    size_t dim = (size_t)(info.shape[1]);

    if (n_samples == 0) throw py::value_error("data must have at least one sample");
    if (dim == 0 || info.shape[2] != info.shape[1]) throw py::value_error("data must contain square matrices");

    if ( (n + 3) / 2 >= max_order() ) throw py::value_error("Current GL table is not large enough for given n");

    CArray<NumT> out({info.shape[1], info.shape[2]});
    auto out_info = out.request();

    Magnus::MatrixSpan<NumT> data_view((NumT*)(info.ptr), dim, n_samples );
    Magnus::MatrixView<NumT> out_view((NumT*)(out_info.ptr), dim);

    {
        py::gil_scoped_release release;
        ::magnus_one<StandardPolicy<NumT>>(
            out_view,
            n,
            data_view,
            t0, tf
        );
    }

    return out;
}

template <class NumT>
CArray<NumT> magnus_many( size_t n, CArray_Coercible<NumT> data, double t0, double tf ) {
    auto info = data.request();
    
    if ( info.ndim != 3 ) throw py::value_error("data must have size (samples, dim, dim)");

    size_t n_samples = (size_t)(info.shape[0]);
    size_t dim = (size_t)(info.shape[1]);

    if (n_samples == 0) throw py::value_error("data must have at least one sample");
    if (dim == 0 || info.shape[2] != info.shape[1]) throw py::value_error("data must contain square matrices");

    if ( (n + 3) / 2 >= max_order() ) throw py::value_error("Current GL table is not large enough for given n");

    CArray<NumT> out({n, dim, dim});
    auto out_info = out.request();

    Magnus::MatrixSpan<NumT> data_view((NumT*)(info.ptr), dim, n_samples );
    Magnus::MatrixSpan<NumT> out_view((NumT*)(out_info.ptr), dim, n);

    {
        py::gil_scoped_release release;
        ::magnus_many<StandardPolicy<NumT>>(
            out_view,
            data_view,
            t0, tf
        );
    }

    return out;
}

template <class NumT>
CArray<NumT> magnus_sum( size_t n, CArray_Coercible<NumT> data, double t0, double tf ) {
    auto info = data.request();
    
    if ( info.ndim != 3 ) throw py::value_error("data must have size (samples, dim, dim)");

    size_t n_samples = (size_t)(info.shape[0]);
    size_t dim = (size_t)(info.shape[1]);

    if (n_samples == 0) throw py::value_error("data must have at least one sample");
    if (dim == 0 || info.shape[2] != info.shape[1]) throw py::value_error("data must contain square matrices");

    if ( (n + 3) / 2 >= max_order() ) throw py::value_error("Current GL table is not large enough for given n");

    CArray<NumT> out({dim, dim});
    auto out_info = out.request();

    Magnus::MatrixSpan<NumT> data_view((NumT*)(info.ptr), dim, n_samples );
    Magnus::MatrixView<NumT> out_view((NumT*)(out_info.ptr), dim);

    {
        py::gil_scoped_release release;
        ::magnus_sum<StandardPolicy<NumT>>(
            out_view,
            n,
            data_view,
            t0, tf
        );
    }

    return out;
}

}

PYBIND11_MODULE(_core, m) {
    GLTable::update(std::make_shared<StaticTable>(_binary_gl_nodes_bin_start));
    m.doc() = "Python bindings for the magnus C++ library";
    m.def("max_order", &Magnus::detail::max_order, "Return the maximum built-in Gauss-Legendre order");
    m.def(
        "magnus_one", 
        &Magnus::detail::magnus_one<double>, 
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        "Compute exactly the nth term of the Magnus term. Saves on some space and runtime."
    );
    m.def(
        "magnus_many", 
        &Magnus::detail::magnus_many<double>, 
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        "Compute all terms from 1 to n of the Magnus expansion in one run, reusing intermediate steps."
    );
    m.def(
        "magnus_sum", 
        &Magnus::detail::magnus_sum<double>, 
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        "Compute the sum of Magnus expansion terms from 1 to n."
    );

}
