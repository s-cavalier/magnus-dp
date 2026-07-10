#ifndef __GRAD_HPP__
#define __GRAD_HPP__
#include "graddata.hpp"
#include "magnus.hpp"
#include "integrate.hpp"
#include "util.hpp"
#include <optional>

namespace Magnus::VJP {

    // Special magnus.hpp-style kernel for just getting intermediate, VJP data.
    // we don't save any fwd values, hence the name `none`. But, we do store vjp_data.
    template <Integrator Int>
    void none(
        size_t n,
        typename Int::matrix_span_t& A,
        double t0, double tf,
        VJP::Data<typename Int::numeric_t>& vjp_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using namespace Magnus;
        using NumT = typename Int::numeric_t;
        using MatrixViewT = typename Int::matrix_t;
        using MatrixSpanT = typename Int::matrix_span_t;

        size_t mat_dim = A.mat_dim();
        size_t matrix_size = mat_dim * mat_dim;
        size_t sample_len = A.length();
        size_t total_data_size = matrix_size * sample_len;
        double dt = (tf - t0) / (sample_len - 1);

        Int integrator( mat_dim, alloc );

        if ( n == 1 ) return;

        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        auto Y_data = allocate_unique<NumT>(total_data_size, alloc);
        auto total_data = allocate_unique<NumT>(matrix_size, alloc);
        MatrixSpanT Y( Y_data.get(), mat_dim, sample_len );
        MatrixViewT total_copy( total_data.get(), mat_dim );

        MatrixViewT temp = integrator.borrow_scratch();

        for ( size_t q = 0; q < view.order(); ++q ) {
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;

            Y.copy_from(A);

            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                vjp_data.record_prefix(q, k, Y);

                MatrixViewT total = Y[ sample_len - 1 ];
                total_copy.copy_from(total);
                Y.sample_update(A, total_copy, shift, temp);
            }
        }
    }

    template <Integrator Int>
    void one(
        typename Int::matrix_span_t& dA,
        typename Int::matrix_t& dOut,
        size_t n,
        typename Int::matrix_span_t& A,
        double t0,
        double tf,
        const VJP::Data<typename Int::numeric_t>* carry_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using NumT = typename Int::numeric_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        using DataOwnerT = decltype(allocate_unique<NumT>(size_t{}, alloc));
        
        size_t dim = A.mat_dim();
        size_t samples = A.length();
        size_t matrix_size = dim * dim;
        size_t span_size = samples * matrix_size;
        double dt = (tf - t0) / (samples - 1);

        Int integrator(dim, alloc);

        if (n == 1) {
            integrator.sum_vjp(dA, dOut, dt);
            return;
        };

        dA.zero();

        std::optional<Data<NumT>> fwd_data_tmp;
        std::optional<DataOwnerT> local_data;

        if ( carry_data ) fwd_data_tmp.emplace(*carry_data);
        else {
            size_t data_size = gl_max_n(n) * total_orders(n) * samples * matrix_size;

            local_data.emplace(allocate_unique<NumT>(data_size, alloc));
            fwd_data_tmp.emplace(local_data->get(), n, samples, dim);

            none<Int>(
                n,
                A,
                t0,
                tf,
                *fwd_data_tmp,
                alloc
            );
        }

        auto& fwd_data = *fwd_data_tmp;

        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order((n + 3) / 2);

        auto barY_data = allocate_unique<NumT>(span_size, alloc);
        SpanT barY(barY_data.get(), dim, samples);

        MatrixT tmp = integrator.borrow_scratch();

        for (size_t q = 0; q < view.order(); ++q) {
            auto [w_q, x_q] = view[q];

            double shift = x_q - 1.0;
            double dscale = dt * w_q;

            integrator.sum_vjp(barY, dOut, dscale);

            for (size_t k = n; k > 1; --k) {
                SpanT P = fwd_data.template prefix<MatPolicyT>(q, k);

                barY.sample_update_vjp(
                    dA,
                    A,
                    P,
                    shift,
                    tmp
                );

                integrator.prefix_vjp(barY, dt);
            }

            // Reverse of the initial Y.copy_from(A) at the start of this q path.
            dA.add(barY);
        }
    }

    template <Integrator Int>
    void many(
        typename Int::matrix_span_t& dA,
        typename Int::matrix_span_t& dOut,
        typename Int::matrix_span_t& A,
        double t0,
        double tf,
        const VJP::Data<typename Int::numeric_t>* carry_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using NumT = typename Int::numeric_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        using DataOwnerT = decltype(allocate_unique<NumT>(size_t{}, alloc));

        size_t n = dOut.length();
        size_t dim = A.mat_dim();
        size_t samples = A.length();
        size_t matrix_size = dim * dim;
        size_t span_size = samples * matrix_size;
        double dt = (tf - t0) / (samples - 1);

        Int integrator(dim, alloc);

        MatrixT first_dOut = dOut[0];
        integrator.sum_vjp(dA, first_dOut, dt);

        if (n == 1) return;

        std::optional<Data<NumT>> fwd_data_tmp;
        std::optional<DataOwnerT> local_data;

        if ( carry_data ) fwd_data_tmp.emplace(*carry_data);
        else {
            size_t data_size = gl_max_n(n) * total_orders(n) * samples * matrix_size;

            local_data.emplace(allocate_unique<NumT>(data_size, alloc));
            fwd_data_tmp.emplace(local_data->get(), n, samples, dim);

            none<Int>(
                n,
                A,
                t0,
                tf,
                *fwd_data_tmp,
                alloc
            );
        }

        auto& fwd_data = *fwd_data_tmp;

        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order((n + 3) / 2);

        auto barY_data = allocate_unique<NumT>(span_size, alloc);
        SpanT barY(barY_data.get(), dim, samples);

        MatrixT tmp = integrator.borrow_scratch();

        for (size_t q = 0; q < view.order(); ++q) {
            auto [w_q, x_q] = view[q];

            double shift = x_q - 1.0;
            double dscale = dt * w_q;

            MatrixT last_dOut = dOut[n - 1];
            integrator.sum_vjp(barY, last_dOut, dscale);

            for (size_t k = n; k > 2; --k) {
                SpanT P = fwd_data.template prefix<MatPolicyT>(q, k);

                barY.sample_update_vjp(
                    dA,
                    A,
                    P,
                    shift,
                    tmp
                );

                integrator.prefix_vjp(barY, dt);

                MatrixT direct_dOut = dOut[k - 2];
                integrator.sum_vjp_add(barY, direct_dOut, dscale);
            }

            SpanT P = fwd_data.template prefix<MatPolicyT>(q, 2);

            barY.sample_update_vjp(
                dA,
                A,
                P,
                shift,
                tmp
            );

            integrator.prefix_vjp(barY, dt);

            dA.add(barY);
        }

    }

    template <Integrator Int>
    void sum(
        typename Int::matrix_span_t& dA,
        typename Int::matrix_t& dOut,
        size_t n,
        typename Int::matrix_span_t& A,
        double t0,
        double tf,
        const VJP::Data<typename Int::numeric_t>* carry_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using NumT = typename Int::numeric_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        using DataOwnerT = decltype(allocate_unique<NumT>(size_t{}, alloc));

        size_t dim = A.mat_dim();
        size_t samples = A.length();
        size_t matrix_size = dim * dim;
        size_t span_size = samples * matrix_size;
        double dt = (tf - t0) / (samples - 1);

        Int integrator(dim, alloc);

        integrator.sum_vjp(dA, dOut, dt);

        if (n == 1) return;

        std::optional<Data<NumT>> fwd_data_tmp;
        std::optional<DataOwnerT> local_data;

        if ( carry_data ) fwd_data_tmp.emplace(*carry_data);
        else {
            size_t data_size = gl_max_n(n) * total_orders(n) * samples * matrix_size;

            local_data.emplace(allocate_unique<NumT>(data_size, alloc));
            fwd_data_tmp.emplace(local_data->get(), n, samples, dim);

            none<Int>(
                n,
                A,
                t0,
                tf,
                *fwd_data_tmp,
                alloc
            );
        }

        auto& fwd_data = *fwd_data_tmp;

        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order((n + 3) / 2);

        auto barY_data = allocate_unique<NumT>(span_size, alloc);
        SpanT barY(barY_data.get(), dim, samples);

        MatrixT tmp = integrator.borrow_scratch();

        for (size_t q = 0; q < view.order(); ++q) {
            auto [w_q, x_q] = view[q];

            double shift = x_q - 1.0;
            double dscale = dt * w_q;

            integrator.sum_vjp(barY, dOut, dscale);

            for (size_t k = n; k > 2; --k) {
                SpanT P = fwd_data.template prefix<MatPolicyT>(q, k);

                barY.sample_update_vjp(
                    dA,
                    A,
                    P,
                    shift,
                    tmp
                );

                integrator.prefix_vjp(barY, dt);

                integrator.sum_vjp_add(barY, dOut, dscale);
            }

            SpanT P = fwd_data.template prefix<MatPolicyT>(q, 2);

            barY.sample_update_vjp(
                dA,
                A,
                P,
                shift,
                tmp
            );

            integrator.prefix_vjp(barY, dt);

            dA.add(barY);
        }

    }


}


#endif
