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
CArray<NumT> magnus( size_t n, CArray_Coercible<NumT> data, NumT t0, NumT tf ) {
    auto info = data.request();
    
    if ( info.ndim != 3 ) throw py::value_error("data must have size (samples, dim, dim)");

    size_t n_samples = (size_t)(info.shape[0]);
    size_t dim = (size_t)(info.shape[1]);

    if (n_samples == 0) throw py::value_error("data must have at least one sample");
    if (dim == 0 || info.shape[2] != info.shape[1]) throw py::value_error("data must contain square matrices");

    if ( n >= max_order() ) throw py::value_error("Current GL table is not large enough for given n");

    CArray<NumT> out({info.shape[1], info.shape[2]});
    auto out_info = out.request();

    NumT* data_ptr = (NumT*)(info.ptr);
    Magnus::MatrixView<NumT> out_view((NumT*)(out_info.ptr), dim);

    {
        py::gil_scoped_release release;
        ::magnus(
            out_view,
            n,
            data_ptr,
            t0, tf, n_samples
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
        "magnus_f64", 
        &Magnus::detail::magnus<double>, 
        py::arg("n"),
        py::arg("data"),
        py::arg("t0"),
        py::arg("tf"),
        "Compute Magnus expansion for fp64 data using evenly spaced samples."
    );
}
