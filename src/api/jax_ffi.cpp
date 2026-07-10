#include "composer/backends.hpp"

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

template <Numeric NumT>
ffi::Error matrix_impl(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    bool record_vjp,
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
        matrix_backend,
        integrator,
        record_vjp
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error matrix_impl_vjp(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    size_t samples,
    size_t dim,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> vjp_out
) {
    run_raw<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        vjp_out->typed_data<NumT>(),
        samples,
        dim,
        t0,
        tf,
        op,
        matrix_backend,
        integrator,
        true
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error spacecurve_impl(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view integrator,
    bool record_vjp,
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
        integrator,
        record_vjp
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error spacecurve_impl_vjp(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view integrator,
    size_t samples,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> vjp_out
) {
    run_spacecurve_raw<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        vjp_out->typed_data<NumT>(),
        samples,
        t0,
        tf,
        op,
        integrator,
        true
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
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    bool record_vjp,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");

        detail::MatrixInputShape input_shape = detail::matrix_input_shape(n, data.dimensions());
        detail::validate_matrix_output_shape(op_attr, n, input_shape.dim, out->dimensions());

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            matrix_impl,
            n,
            op_attr,
            t0,
            tf,
            matrix_backend,
            integrator,
            record_vjp,
            input_shape.samples,
            input_shape.dim,
            data,
            out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus matrix FFI error");
    }
}

ffi::Error matrix_dispatch_vjp(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> vjp_out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");
        if (data.element_type() != vjp_out->element_type()) return ffi::Error::InvalidArgument("input and vjp output dtypes must match");

        detail::MatrixInputShape input_shape = detail::matrix_input_shape(n, data.dimensions());
        detail::validate_matrix_output_shape(op_attr, n, input_shape.dim, out->dimensions());

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            matrix_impl_vjp,
            n,
            op_attr,
            t0,
            tf,
            matrix_backend,
            integrator,
            input_shape.samples,
            input_shape.dim,
            data,
            out,
            vjp_out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus matrix VJP FFI error");
    }
}

ffi::Error spacecurve_dispatch(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view integrator,
    bool record_vjp,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");

        detail::SpaceCurveInputShape input_shape = detail::spacecurve_input_shape(n, data.dimensions());
        detail::validate_spacecurve_output_shape(op_attr, n, out->dimensions());

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            spacecurve_impl,
            n,
            op_attr,
            t0,
            tf,
            integrator,
            record_vjp,
            input_shape.samples,
            data,
            out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus spacecurve FFI error");
    }
}

ffi::Error spacecurve_dispatch_vjp(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> vjp_out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");
        if (data.element_type() != vjp_out->element_type()) return ffi::Error::InvalidArgument("input and vjp output dtypes must match");

        detail::SpaceCurveInputShape input_shape = detail::spacecurve_input_shape(n, data.dimensions());
        detail::validate_spacecurve_output_shape(op_attr, n, out->dimensions());

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            spacecurve_impl_vjp,
            n,
            op_attr,
            t0,
            tf,
            integrator,
            input_shape.samples,
            data,
            out,
            vjp_out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus spacecurve VJP FFI error");
    }
}

#undef MAGNUS_ELEMENT_TYPE_DISPATCH

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_matrix,
    matrix_dispatch,
    ffi::Ffi::Bind()
        .Attr<size_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("matrix_backend")
        .Attr<std::string_view>("integrator")
        .Attr<bool>("record_vjp")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_matrix_vjp,
    matrix_dispatch_vjp,
    ffi::Ffi::Bind()
        .Attr<size_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("matrix_backend")
        .Attr<std::string_view>("integrator")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_spacecurve,
    spacecurve_dispatch,
    ffi::Ffi::Bind()
        .Attr<size_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("integrator")
        .Attr<bool>("record_vjp")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_spacecurve_vjp,
    spacecurve_dispatch_vjp,
    ffi::Ffi::Bind()
        .Attr<size_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("integrator")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
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
    targets["magnus_matrix_vjp"] = encapsulate_ffi_handler(magnus_matrix_vjp);
    targets["magnus_spacecurve"] = encapsulate_ffi_handler(magnus_spacecurve);
    targets["magnus_spacecurve_vjp"] = encapsulate_ffi_handler(magnus_spacecurve_vjp);
    return targets;
}

}

void bind(py::module_& m) {
    m.def( "registrations", &registrations, "Return JAX FFI CPU target registrations." );
}

}
