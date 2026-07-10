#ifndef __BACKEND_COMPOSER_HPP__
#define __BACKEND_COMPOSER_HPP__
#include "dispatch.hpp"
#include "integration/backends.hpp"
#include "linalg/backends.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace Magnus {

    void initialize_default_gl_table();

    std::unique_ptr<KernelPlan> make_plan(Params& p, size_t num_idx, size_t mat_idx, size_t int_idx, bool vjp_record, Dispatch::KernelOp op);

    namespace detail {

        struct MatrixInputShape {
            size_t samples;
            size_t dim;
        };

        struct SpaceCurveInputShape {
            size_t samples;
        };

        template <class T>
        inline size_t checked_shape_extent(T value, std::string_view name) {
            static_assert(std::is_integral_v<T>, "shape extents must be integral");

            if constexpr (std::is_signed_v<T>) {
                if (value < 0) throw std::invalid_argument(std::string(name) + " must be non-negative");
            }

            using UnsignedT = std::make_unsigned_t<T>;
            constexpr size_t max_size = std::numeric_limits<size_t>::max();
            if (static_cast<UnsignedT>(value) > max_size) {
                throw std::invalid_argument(std::string(name) + " is too large");
            }

            return static_cast<size_t>(value);
        }

        inline size_t checked_matrix_entry_count(size_t samples, size_t dim) {
            if (dim != 0 && dim > std::numeric_limits<size_t>::max() / dim) throw std::invalid_argument("matrix dimension is too large");
            size_t matrix_size = dim * dim;
            if (samples != 0 && matrix_size > std::numeric_limits<size_t>::max() / samples) throw std::invalid_argument("input shape is too large");
            return samples * matrix_size;
        }

        inline void checked_spacecurve_entry_count(size_t samples) {
            if (samples > std::numeric_limits<size_t>::max() / 3) throw std::invalid_argument("input shape is too large");
        }

        inline size_t output_count(Dispatch::KernelOp op, size_t n) {
            switch (op) {
                case Dispatch::KernelOp::ONE:
                case Dispatch::KernelOp::SUM:
                    return 1;
                case Dispatch::KernelOp::MANY:
                    return n;
            }

            throw std::invalid_argument("invalid kernel operation");
        }

        inline size_t output_count(std::string_view op, size_t n) {
            return output_count(Dispatch::op_from_str(op), n);
        }

        inline std::vector<size_t> matrix_output_shape(Dispatch::KernelOp op, size_t n, size_t dim) {
            if (op == Dispatch::KernelOp::MANY) return {n, dim, dim};

            if (op == Dispatch::KernelOp::ONE || op == Dispatch::KernelOp::SUM) return {dim, dim};

            throw std::invalid_argument("invalid kernel operation");
        }

        inline std::vector<size_t> matrix_output_shape(std::string_view op, size_t n, size_t dim) {
            return matrix_output_shape(Dispatch::op_from_str(op), n, dim);
        }

        inline std::vector<size_t> spacecurve_output_shape(Dispatch::KernelOp op, size_t n) {
            if (op == Dispatch::KernelOp::MANY) return {n, 3};

            if (op == Dispatch::KernelOp::ONE || op == Dispatch::KernelOp::SUM) return {3};

            throw std::invalid_argument("invalid kernel operation");
        }

        inline std::vector<size_t> spacecurve_output_shape(std::string_view op, size_t n) {
            return spacecurve_output_shape(Dispatch::op_from_str(op), n);
        }

        template <class Shape>
        inline MatrixInputShape matrix_input_shape(size_t n, const Shape& shape) {
            if (n == 0) throw std::invalid_argument("n must be at least 1");
            if (shape.size() != 3) throw std::invalid_argument("data must have shape (samples, dim, dim)");

            const size_t samples = checked_shape_extent(shape[0], "sample count");
            const size_t rows = checked_shape_extent(shape[1], "matrix row count");
            const size_t cols = checked_shape_extent(shape[2], "matrix column count");

            if (samples < 2) throw std::invalid_argument("data must contain at least two samples");
            if (rows == 0 || cols != rows) throw std::invalid_argument("data must contain square matrices");

            checked_matrix_entry_count(samples, rows);
            return {samples, rows};
        }

        template <class Shape>
        inline SpaceCurveInputShape spacecurve_input_shape(size_t n, const Shape& shape) {
            if (n == 0) throw std::invalid_argument("n must be at least 1");
            if (shape.size() != 2) throw std::invalid_argument("data must have shape (samples, 3)");

            const size_t samples = checked_shape_extent(shape[0], "sample count");
            const size_t width = checked_shape_extent(shape[1], "spacecurve width");

            if (samples < 2) throw std::invalid_argument("data must contain at least two samples");
            if (width != 3) throw std::invalid_argument("data must have shape (samples, 3)");

            checked_spacecurve_entry_count(samples);
            return {samples};
        }

        template <class Shape>
        inline void validate_shape(const Shape& actual, const std::vector<size_t>& expected, std::string_view message) {
            if (actual.size() != expected.size()) throw std::invalid_argument(std::string(message));

            for (size_t i = 0; i < expected.size(); ++i) {
                if (checked_shape_extent(actual[i], "output shape extent") != expected[i]) {
                    throw std::invalid_argument(std::string(message));
                }
            }
        }

        template <class Shape>
        inline void validate_matrix_output_shape(Dispatch::KernelOp op, size_t n, size_t dim, const Shape& actual) {
            const std::vector<size_t> expected = matrix_output_shape(op, n, dim);
            validate_shape(actual, expected, op == Dispatch::KernelOp::MANY ? "many output must have shape (n, dim, dim)" : "one/sum output must have shape (dim, dim)");
        }

        template <class Shape>
        inline void validate_matrix_output_shape(std::string_view op, size_t n, size_t dim, const Shape& actual) {
            validate_matrix_output_shape(Dispatch::op_from_str(op), n, dim, actual);
        }

        template <class Shape>
        inline void validate_spacecurve_output_shape(Dispatch::KernelOp op, size_t n, const Shape& actual) {
            const std::vector<size_t> expected = spacecurve_output_shape(op, n);
            validate_shape(actual, expected, op == Dispatch::KernelOp::MANY ? "many spacecurve output must have shape (n, 3)" : "one/sum spacecurve output must have shape (3,)");
        }

        template <class Shape>
        inline void validate_spacecurve_output_shape(std::string_view op, size_t n, const Shape& actual) {
            validate_spacecurve_output_shape(Dispatch::op_from_str(op), n, actual);
        }

        inline void validate_raw_buffers(const void* in, const void* out) {
            if (in == nullptr) throw std::invalid_argument("input buffer must not be null");
            if (out == nullptr) throw std::invalid_argument("output buffer must not be null");
        }

        template <Numeric NumT>
        void run_raw_mutable(
            size_t n,
            NumT* in,
            NumT* out,
            NumT* vjp_data,
            size_t samples,
            size_t dim,
            double t0,
            double tf,
            Dispatch::KernelOp op,
            size_t mat_idx,
            size_t int_idx,
            bool record_vjp
        ) {
            validate_raw_buffers(in, out);
            matrix_input_shape(n, std::array<size_t, 3>{samples, dim, dim});
            initialize_default_gl_table();

            Params params{
                Params::num_data<NumT>{in, out, vjp_data},
                n,
                dim,
                samples,
                t0,
                tf
            };

            size_t num_idx = NumBackends::resolve(type_name_v<NumT>);

            std::unique_ptr<KernelPlan> plan = make_plan(params, num_idx, mat_idx, int_idx, record_vjp, op);
            plan->run();
        }

    }

    template <Numeric NumT>
    void run_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        NumT* vjp_data,
        size_t samples,
        size_t dim,
        double t0,
        double tf,
        Dispatch::KernelOp op,
        size_t mat_idx,
        size_t int_idx,
        bool record_vjp = false
    ) {
        detail::validate_raw_buffers(in, out);
        detail::matrix_input_shape(n, std::array<size_t, 3>{samples, dim, dim});

        size_t entry_count = detail::checked_matrix_entry_count(samples, dim);
        std::vector<NumT> input_copy(entry_count);
        std::copy_n(in, entry_count, input_copy.data());

        detail::run_raw_mutable(
            n,
            input_copy.data(),
            out,
            vjp_data,
            samples,
            dim,
            t0,
            tf,
            op,
            mat_idx,
            int_idx,
            record_vjp
        );
    }

    template <Numeric NumT>
    void run_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        size_t samples,
        size_t dim,
        double t0,
        double tf,
        std::string_view op,
        std::string_view matrix_backend,
        std::string_view integrator,
        bool record_vjp = false
    ) {
        run_raw(
            n,
            in,
            out,
            static_cast<NumT*>(nullptr),
            samples,
            dim,
            t0,
            tf,
            Dispatch::op_from_str(op),
            MatrixBackends::resolve(matrix_backend),
            IntegratorBackends::resolve(integrator),
            record_vjp
        );
    }

    template <Numeric NumT>
    void run_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        NumT* vjp_data,
        size_t samples,
        size_t dim,
        double t0,
        double tf,
        std::string_view op,
        std::string_view matrix_backend,
        std::string_view integrator,
        bool record_vjp = false
    ) {
        run_raw(
            n,
            in,
            out,
            vjp_data,
            samples,
            dim,
            t0,
            tf,
            Dispatch::op_from_str(op),
            MatrixBackends::resolve(matrix_backend),
            IntegratorBackends::resolve(integrator),
            record_vjp
        );
    }

    template <Numeric NumT>
    void run_spacecurve_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        NumT* vjp_data,
        size_t samples,
        double t0,
        double tf,
        Dispatch::KernelOp op,
        size_t int_idx,
        bool record_vjp = false
    ) {
        detail::validate_raw_buffers(in, out);
        detail::spacecurve_input_shape(n, std::array<size_t, 2>{samples, 3});

        size_t result_count = detail::output_count(op, n);
        if (result_count > std::numeric_limits<size_t>::max() / 4) throw std::invalid_argument("output shape is too large");

        std::vector<NumT> padded(samples * 4);
        for (size_t i = 0; i < samples; ++i) {
            padded[4 * i] = NumT{0};
            padded[4 * i + 1] = in[3 * i];
            padded[4 * i + 2] = in[3 * i + 1];
            padded[4 * i + 3] = in[3 * i + 2];
        }

        std::vector<NumT> padded_out(result_count * 4);
        detail::run_raw_mutable(
            n,
            padded.data(),
            padded_out.data(),
            vjp_data,
            samples,
            2,
            t0,
            tf,
            op,
            MatrixBackends::resolve("SpaceCurve"),
            int_idx,
            record_vjp
        );

        for (size_t i = 0; i < result_count; ++i) {
            out[3 * i] = padded_out[4 * i + 1];
            out[3 * i + 1] = padded_out[4 * i + 2];
            out[3 * i + 2] = padded_out[4 * i + 3];
        }
    }

    template <Numeric NumT>
    void run_spacecurve_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        size_t samples,
        double t0,
        double tf,
        std::string_view op,
        std::string_view integrator,
        bool record_vjp = false
    ) {
        run_spacecurve_raw(
            n,
            in,
            out,
            static_cast<NumT*>(nullptr),
            samples,
            t0,
            tf,
            Dispatch::op_from_str(op),
            IntegratorBackends::resolve(integrator),
            record_vjp
        );
    }

    template <Numeric NumT>
    void run_spacecurve_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        NumT* vjp_data,
        size_t samples,
        double t0,
        double tf,
        std::string_view op,
        std::string_view integrator,
        bool record_vjp = false
    ) {
        run_spacecurve_raw(
            n,
            in,
            out,
            vjp_data,
            samples,
            t0,
            tf,
            Dispatch::op_from_str(op),
            IntegratorBackends::resolve(integrator),
            record_vjp
        );
    }

}

#endif
