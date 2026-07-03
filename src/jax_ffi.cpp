#include "backendcomposer.hpp"

#include <cstdint>
#include <exception>
#include <string_view>
#include <type_traits>

#include <pybind11/pybind11.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace Magnus::JaxFfi {

namespace {

ffi::Error validate_matrix_output_shape(const ffi::AnyBuffer::Dimensions& dims, Dispatch::KernelOp op, size_t n, size_t dim) {
    if (op == Dispatch::KernelOp::MANY) {
        if (
            dims.size() != 3
            || static_cast<size_t>(dims[0]) != n
            || static_cast<size_t>(dims[1]) != dim
            || static_cast<size_t>(dims[2]) != dim
        ) {
            return ffi::Error::InvalidArgument("many output must have shape (n, dim, dim)");
        }

        return ffi::Error::Success();
    }

    if (dims.size() != 2 || static_cast<size_t>(dims[0]) != dim || static_cast<size_t>(dims[1]) != dim) {
        return ffi::Error::InvalidArgument("one/sum output must have shape (dim, dim)");
    }

    return ffi::Error::Success();
}

ffi::Error validate_spacecurve_output_shape(const ffi::AnyBuffer::Dimensions& dims, Dispatch::KernelOp op, size_t n) {
    if (op == Dispatch::KernelOp::MANY) {
        if (dims.size() != 2 || static_cast<size_t>(dims[0]) != n || static_cast<size_t>(dims[1]) != 3) {
            return ffi::Error::InvalidArgument("many spacecurve output must have shape (n, 3)");
        }

        return ffi::Error::Success();
    }

    if (dims.size() != 1 || static_cast<size_t>(dims[0]) != 3) {
        return ffi::Error::InvalidArgument("one/sum spacecurve output must have shape (3,)");
    }

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error matrix_impl(
    size_t n,
    Dispatch::KernelOp op,
    double t0,
    double tf,
    size_t matrix_backend_idx,
    size_t integrator_idx,
    size_t samples,
    size_t dim,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    run_raw<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        samples,
        dim,
        t0,
        tf,
        op,
        matrix_backend_idx,
        integrator_idx
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error spacecurve_impl(
    size_t n,
    Dispatch::KernelOp op,
    double t0,
    double tf,
    size_t integrator_idx,
    size_t samples,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    run_spacecurve_raw<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        samples,
        t0,
        tf,
        op,
        integrator_idx
    );

    return ffi::Error::Success();
}

#define MAGNUS_ELEMENT_TYPE_DISPATCH(element_type, fn, ...) \
    switch (element_type) {                                 \
        case ffi::F32:                                      \
            return fn<f32>(__VA_ARGS__);                    \
        case ffi::F64:                                      \
            return fn<f64>(__VA_ARGS__);                    \
        case ffi::C64:                                      \
            return fn<c32>(__VA_ARGS__);                    \
        case ffi::C128:                                     \
            return fn<c64>(__VA_ARGS__);                    \
        default:                                            \
            return ffi::Error::InvalidArgument("unsupported data dtype"); \
    }

ffi::Error matrix_dispatch(
    std::uint64_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");

        Dispatch::KernelOp op = Dispatch::op_from_str(op_attr);
        size_t matrix_backend_idx = MatrixBackends::resolve(matrix_backend);
        size_t integrator_idx = IntegratorBackends::resolve(integrator);

        ffi::AnyBuffer::Dimensions dims = data.dimensions();
        if (dims.size() != 3) return ffi::Error::InvalidArgument("matrix data must have shape (samples, dim, dim)");

        size_t samples = static_cast<size_t>(dims[0]);
        size_t rows = static_cast<size_t>(dims[1]);
        size_t cols = static_cast<size_t>(dims[2]);
        if (rows == 0 || rows != cols) return ffi::Error::InvalidArgument("matrix data must contain square matrices");

        ffi::Error output_error = validate_matrix_output_shape( out->dimensions(), op, n, rows );

        if (output_error.failure()) return output_error;

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            matrix_impl,
            n,
            op,
            t0,
            tf,
            matrix_backend_idx,
            integrator_idx,
            samples,
            rows,
            data,
            out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus matrix FFI error");
    }
}

ffi::Error spacecurve_dispatch(
    std::uint64_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");
        Dispatch::KernelOp op = Dispatch::op_from_str(op_attr);
        size_t integrator_idx = IntegratorBackends::resolve(integrator);

        ffi::AnyBuffer::Dimensions dims = data.dimensions();
        if (dims.size() != 2) return ffi::Error::InvalidArgument( "spacecurve data must have shape (samples, 3)" );

        size_t samples = static_cast<size_t>(dims[0]);
        size_t width = static_cast<size_t>(dims[1]);
        if (width != 3) return ffi::Error::InvalidArgument( "spacecurve data must have shape (samples, 3)" );

        ffi::Error output_error = validate_spacecurve_output_shape( out->dimensions(), op, n);

        if (output_error.failure()) return output_error;

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            spacecurve_impl,
            n,
            op,
            t0,
            tf,
            integrator_idx,
            samples,
            data,
            out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus spacecurve FFI error");
    }
}

#undef MAGNUS_ELEMENT_TYPE_DISPATCH

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_matrix,
    matrix_dispatch,
    ffi::Ffi::Bind()
        .Attr<std::uint64_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("matrix_backend")
        .Attr<std::string_view>("integrator")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_spacecurve,
    spacecurve_dispatch,
    ffi::Ffi::Bind()
        .Attr<std::uint64_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("integrator")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

template <typename T>
py::capsule encapsulate_ffi_handler(T* fn) {
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
        "encapsulated function must be an XLA FFI handler"
    );

    return py::capsule(reinterpret_cast<void*>(fn));
}

py::dict registrations() {
    py::dict targets;
    targets["magnus_matrix"] = encapsulate_ffi_handler(magnus_matrix);
    targets["magnus_spacecurve"] = encapsulate_ffi_handler(magnus_spacecurve);
    return targets;
}

}

void bind(py::module_& m) {
    m.def( "registrations", &registrations, "Return JAX FFI CPU target registrations." );
}

}
