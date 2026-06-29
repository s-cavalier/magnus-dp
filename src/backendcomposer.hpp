#ifndef __BACKEND_COMPOSER_HPP__
#define __BACKEND_COMPOSER_HPP__
#include "dispatch.hpp"
#include "integratebackends.hpp"
#include "matrixbackends.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace Magnus {

    void initialize_default_gl_table();

    std::unique_ptr<KernelPlan> make_plan(Params& p, size_t num_idx, size_t mat_idx, size_t int_idx);

    namespace detail {

        inline size_t checked_matrix_entry_count(size_t samples, size_t dim) {
            if (dim != 0 && dim > std::numeric_limits<size_t>::max() / dim) throw std::invalid_argument("matrix dimension is too large");
            size_t matrix_size = dim * dim;
            if (samples != 0 && matrix_size > std::numeric_limits<size_t>::max() / samples) throw std::invalid_argument("input shape is too large");
            return samples * matrix_size;
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

        inline void validate_raw_input_shape(
            size_t n,
            const void* in,
            const void* out,
            size_t samples,
            size_t dim
        ) {
            if (n == 0) throw std::invalid_argument("n must be at least 1");
            if (in == nullptr) throw std::invalid_argument("input buffer must not be null");
            if (out == nullptr) throw std::invalid_argument("output buffer must not be null");
            if (samples < 2) throw std::invalid_argument("data must contain at least two samples");
            if (dim == 0) throw std::invalid_argument("data must contain square matrices");
        }

        inline void validate_raw_spacecurve_input_shape(
            size_t n,
            const void* in,
            const void* out,
            size_t samples
        ) {
            if (n == 0) throw std::invalid_argument("n must be at least 1");
            if (in == nullptr) throw std::invalid_argument("input buffer must not be null");
            if (out == nullptr) throw std::invalid_argument("output buffer must not be null");
            if (samples < 2) throw std::invalid_argument("data must contain at least two samples");
        }

        template <Numeric NumT>
        void run_raw_mutable(
            size_t n,
            NumT* in,
            NumT* out,
            size_t samples,
            size_t dim,
            double t0,
            double tf,
            Dispatch::KernelOp op,
            size_t mat_idx,
            size_t int_idx
        ) {
            validate_raw_input_shape(n, in, out, samples, dim);
            initialize_default_gl_table();

            Params params{
                Params::num_data<NumT>{in, out},
                n,
                dim,
                samples,
                t0,
                tf
            };

            size_t num_idx = NumBackends::resolve(type_name_v<NumT>);

            std::unique_ptr<KernelPlan> plan = make_plan(params, num_idx, mat_idx, int_idx);
            plan->run(op);
        }

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
        Dispatch::KernelOp op,
        size_t mat_idx,
        size_t int_idx
    ) {
        detail::validate_raw_input_shape(n, in, out, samples, dim);

        size_t entry_count = detail::checked_matrix_entry_count(samples, dim);
        std::vector<NumT> input_copy(entry_count);
        std::copy_n(in, entry_count, input_copy.data());

        detail::run_raw_mutable(
            n,
            input_copy.data(),
            out,
            samples,
            dim,
            t0,
            tf,
            op,
            mat_idx,
            int_idx
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
        Dispatch::KernelOp op,
        std::string_view matrix_backend,
        std::string_view integrator
    ) {
        run_raw(
            n,
            in,
            out,
            samples,
            dim,
            t0,
            tf,
            op,
            MatrixBackends::resolve(matrix_backend),
            IntegratorBackends::resolve(integrator)
        );
    }

    template <Numeric NumT>
    void run_spacecurve_raw(
        size_t n,
        const NumT* in,
        NumT* out,
        size_t samples,
        double t0,
        double tf,
        Dispatch::KernelOp op,
        size_t int_idx
    ) {
        detail::validate_raw_spacecurve_input_shape(n, in, out, samples);

        if (samples > std::numeric_limits<size_t>::max() / 4) throw std::invalid_argument("input shape is too large");

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
            samples,
            2,
            t0,
            tf,
            op,
            MatrixBackends::resolve("SpaceCurve"),
            int_idx
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
        Dispatch::KernelOp op,
        std::string_view integrator
    ) {
        run_spacecurve_raw(
            n,
            in,
            out,
            samples,
            t0,
            tf,
            op,
            IntegratorBackends::resolve(integrator)
        );
    }

}

#endif
