#include "composer/backends.hpp"

#include <exception>
#include <optional>
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
    size_t samples,
    size_t dim,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    detail::run_raw_impl<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        nullptr,
        samples,
        dim,
        t0,
        tf,
        Dispatch::op_from_str(op),
        MatrixBackends::resolve(matrix_backend),
        IntegratorBackends::resolve(integrator),
        false
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error matrix_impl_fwd(
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
    ffi::Result<ffi::AnyBuffer> carry_out
) {
    detail::run_raw_impl<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        carry_out->typed_data<NumT>(),
        samples,
        dim,
        t0,
        tf,
        Dispatch::op_from_str(op),
        MatrixBackends::resolve(matrix_backend),
        IntegratorBackends::resolve(integrator),
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
    size_t samples,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out
) {
    detail::run_spacecurve_raw_impl<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        nullptr,
        samples,
        t0,
        tf,
        Dispatch::op_from_str(op),
        IntegratorBackends::resolve(integrator),
        false
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error spacecurve_impl_fwd(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view integrator,
    size_t samples,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> carry_out
) {
    detail::run_spacecurve_raw_impl<NumT>(
        n,
        data.typed_data<NumT>(),
        out->typed_data<NumT>(),
        carry_out->typed_data<NumT>(),
        samples,
        t0,
        tf,
        Dispatch::op_from_str(op),
        IntegratorBackends::resolve(integrator),
        true
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error matrix_impl_bwd(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    size_t samples,
    size_t dim,
    ffi::AnyBuffer data,
    ffi::AnyBuffer cotangent,
    std::optional<ffi::AnyBuffer> carry,
    ffi::Result<ffi::AnyBuffer> out
) {
    detail::run_vjp_raw_impl<NumT>(
        n,
        data.typed_data<NumT>(),
        cotangent.typed_data<NumT>(),
        out->typed_data<NumT>(),
        carry ? carry->typed_data<NumT>() : nullptr,
        samples,
        dim,
        t0,
        tf,
        Dispatch::op_from_str(op),
        MatrixBackends::resolve(matrix_backend),
        IntegratorBackends::resolve(integrator)
    );

    return ffi::Error::Success();
}

template <Numeric NumT>
ffi::Error spacecurve_impl_bwd(
    size_t n,
    std::string_view op,
    double t0,
    double tf,
    std::string_view integrator,
    size_t samples,
    ffi::AnyBuffer data,
    ffi::AnyBuffer cotangent,
    std::optional<ffi::AnyBuffer> carry,
    ffi::Result<ffi::AnyBuffer> out
) {
    detail::run_spacecurve_vjp_raw_impl<NumT>(
        n,
        data.typed_data<NumT>(),
        cotangent.typed_data<NumT>(),
        out->typed_data<NumT>(),
        carry ? carry->typed_data<NumT>() : nullptr,
        samples,
        t0,
        tf,
        Dispatch::op_from_str(op),
        IntegratorBackends::resolve(integrator)
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

ffi::Error matrix_dispatch_fwd(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> carry_out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");
        if (data.element_type() != carry_out->element_type()) return ffi::Error::InvalidArgument("input and carry dtypes must match");

        detail::MatrixInputShape input_shape = detail::matrix_input_shape(n, data.dimensions());
        detail::validate_matrix_output_shape(op_attr, n, input_shape.dim, out->dimensions());
        detail::validate_vjp_data_shape(n, input_shape.samples, input_shape.dim, carry_out->dimensions());

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            matrix_impl_fwd,
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
            carry_out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus matrix forward FFI error");
    }
}

ffi::Error spacecurve_dispatch(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view integrator,
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

ffi::Error matrix_dispatch_bwd(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view matrix_backend,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::AnyBuffer cotangent,
    std::optional<ffi::AnyBuffer> carry,
    ffi::Result<ffi::AnyBuffer> out
) {
    try {
        if (data.element_type() != cotangent.element_type()) return ffi::Error::InvalidArgument("input and cotangent dtypes must match");
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");

        detail::MatrixInputShape input_shape = detail::matrix_input_shape(n, data.dimensions());
        detail::validate_matrix_output_shape(op_attr, n, input_shape.dim, cotangent.dimensions());
        detail::validate_shape(out->dimensions(), {input_shape.samples, input_shape.dim, input_shape.dim}, "matrix VJP output must match the input shape");

        if (carry) {
            if (data.element_type() != carry->element_type()) return ffi::Error::InvalidArgument("input and carry dtypes must match");
            detail::validate_vjp_data_shape(n, input_shape.samples, input_shape.dim, carry->dimensions());
        }

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            matrix_impl_bwd,
            n,
            op_attr,
            t0,
            tf,
            matrix_backend,
            integrator,
            input_shape.samples,
            input_shape.dim,
            data,
            cotangent,
            carry,
            out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus matrix backward FFI error");
    }
}

ffi::Error spacecurve_dispatch_bwd(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::AnyBuffer cotangent,
    std::optional<ffi::AnyBuffer> carry,
    ffi::Result<ffi::AnyBuffer> out
) {
    try {
        if (data.element_type() != cotangent.element_type()) return ffi::Error::InvalidArgument("input and cotangent dtypes must match");
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");

        detail::SpaceCurveInputShape input_shape = detail::spacecurve_input_shape(n, data.dimensions());
        detail::validate_spacecurve_output_shape(op_attr, n, cotangent.dimensions());
        detail::validate_shape(out->dimensions(), {input_shape.samples, size_t{3}}, "SpaceCurve VJP output must match the input shape");

        if (carry) {
            if (data.element_type() != carry->element_type()) return ffi::Error::InvalidArgument("input and carry dtypes must match");
            detail::validate_vjp_data_shape(n, input_shape.samples, 2, carry->dimensions());
        }

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            spacecurve_impl_bwd,
            n,
            op_attr,
            t0,
            tf,
            integrator,
            input_shape.samples,
            data,
            cotangent,
            carry,
            out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus SpaceCurve backward FFI error");
    }
}

ffi::Error spacecurve_dispatch_fwd(
    size_t n,
    std::string_view op_attr,
    double t0,
    double tf,
    std::string_view integrator,
    ffi::AnyBuffer data,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> carry_out
) {
    try {
        if (data.element_type() != out->element_type()) return ffi::Error::InvalidArgument("input and output dtypes must match");
        if (data.element_type() != carry_out->element_type()) return ffi::Error::InvalidArgument("input and carry dtypes must match");

        detail::SpaceCurveInputShape input_shape = detail::spacecurve_input_shape(n, data.dimensions());
        detail::validate_spacecurve_output_shape(op_attr, n, out->dimensions());
        detail::validate_vjp_data_shape(n, input_shape.samples, 2, carry_out->dimensions());

        MAGNUS_ELEMENT_TYPE_DISPATCH(
            data.element_type(),
            spacecurve_impl_fwd,
            n,
            op_attr,
            t0,
            tf,
            integrator,
            input_shape.samples,
            data,
            out,
            carry_out
        );
    } catch (const std::exception& err) {
        return ffi::Error::InvalidArgument(err.what());
    } catch (...) {
        return ffi::Error::InvalidArgument("unknown magnus spacecurve forward FFI error");
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
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_matrix_fwd,
    matrix_dispatch_fwd,
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
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_spacecurve_fwd,
    spacecurve_dispatch_fwd,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_matrix_bwd,
    matrix_dispatch_bwd,
    ffi::Ffi::Bind()
        .Attr<size_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("matrix_backend")
        .Attr<std::string_view>("integrator")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .OptionalArg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    magnus_spacecurve_bwd,
    spacecurve_dispatch_bwd,
    ffi::Ffi::Bind()
        .Attr<size_t>("n")
        .Attr<std::string_view>("op")
        .Attr<double>("t0")
        .Attr<double>("tf")
        .Attr<std::string_view>("integrator")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .OptionalArg<ffi::AnyBuffer>()
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
    targets["magnus_matrix_fwd"] = encapsulate_ffi_handler(magnus_matrix_fwd);
    targets["magnus_spacecurve"] = encapsulate_ffi_handler(magnus_spacecurve);
    targets["magnus_spacecurve_fwd"] = encapsulate_ffi_handler(magnus_spacecurve_fwd);
    targets["magnus_matrix_bwd"] = encapsulate_ffi_handler(magnus_matrix_bwd);
    targets["magnus_spacecurve_bwd"] = encapsulate_ffi_handler(magnus_spacecurve_bwd);
    return targets;
}

}

void bind(py::module_& m) {
    m.def( "registrations", &registrations, "Return JAX FFI CPU target registrations." );
}

}
