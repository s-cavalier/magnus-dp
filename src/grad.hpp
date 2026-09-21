#ifndef __GRAD_HPP__
#define __GRAD_HPP__
#include "graddata.hpp"
#include "magnus.hpp"
#include "integration/integrate.hpp"
#include "util/extra.hpp"
#include "util/gausslegendre.hpp"
#include <optional>

namespace Magnus::VJP {

    // Special magnus.hpp-style kernel for just getting intermediate, VJP data.
    // we don't save any fwd values, hence the name `none`. But, we do store vjp_data.
    template <Integrator Int, class GLIntegrator = GL_forloop>
    void none(
        size_t n,
        typename Int::matrix_span_t& A,
        double t0, double tf,
        VJP::Data<typename Int::numeric_t>& vjp_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using namespace Magnus;
        using MatrixViewT = typename Int::matrix_t;

        size_t mat_dim = A.mat_dim();
        size_t sample_len = A.length();
        double dt = (tf - t0) / (sample_len - 1);

        if ( n == 1 ) return;

        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        size_t worker_count = GLIntegrator::lane_count(view.order());
        std::vector<FwdPathWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        for (size_t i = 0; i < worker_count; ++i) workspaces.emplace_back(mat_dim, sample_len, alloc);

        GLIntegrator::invoke(view.order(), [&](size_t q, int ln){
            auto& ws = workspaces[ln];
            double x_q = view[q].second;
            double shift = x_q - 1;

            ws.Y.copy_from(A);

            for ( size_t k = 2; k <= n; ++k ) {
                ws.integrator.prefix(ws.Y, dt);

                vjp_data.record_prefix(q, k, ws.Y);

                MatrixViewT total = ws.Y[sample_len - 1];
                ws.total_copy.copy_from(total);
                ws.Y.sample_update(A, ws.total_copy, shift, ws.temp);
            }
        });
    }

    template <Integrator Int, class GLIntegrator = GL_forloop>
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
        using AllocatorT = typename Int::allocator_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        using DynMatrixSpanT = DynMatrixSpan<NumT, MatPolicyT, AllocatorT>;
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

            none<Int, GLIntegrator>(
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

        size_t worker_count = GLIntegrator::lane_count(view.order());
        MemoryBuffer buffer(vjp_workspace_buffer_bytes<Int>(worker_count, dim, samples));
        auto buffer_alloc = buffer.template get_allocator<NumT, CACHE_LINE_ALIGNMENT>();
        std::vector<VJPWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        workspaces.emplace_back(dim, samples, dA.data(), alloc);
        for (size_t i = 1; i < worker_count; ++i) {
            NumT* local_dA = buffer_alloc.allocate(span_size);
            workspaces.emplace_back(dim, samples, local_dA, alloc);
            workspaces.back().dA.zero();
        }

        GLIntegrator::invoke(view.order(), [&](size_t q, int ln){
            auto& ws = workspaces[ln];
            auto [w_q, x_q] = view[q];

            double shift = x_q - 1.0;
            double dscale = dt * w_q;

            ws.integrator.sum_vjp(ws.barY, dOut, dscale);

            for (size_t k = n; k > 1; --k) {
                SpanT P = fwd_data.template prefix<MatPolicyT>(q, k);

                ws.barY.sample_update_vjp(
                    ws.dA,
                    A,
                    P,
                    shift,
                    ws.temp
                );

                ws.integrator.prefix_vjp(ws.barY, dt);
            }

            // Reverse of the initial Y.copy_from(A) at the start of this q path.
            ws.dA.add(ws.barY);
        });

        for (size_t i = 1; i < worker_count; ++i) dA.add(workspaces[i].dA);
    }

    template <Integrator Int, class GLIntegrator = GL_forloop>
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
        using AllocatorT = typename Int::allocator_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        using DynMatrixSpanT = DynMatrixSpan<NumT, MatPolicyT, AllocatorT>;
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

            none<Int, GLIntegrator>(
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

        size_t worker_count = GLIntegrator::lane_count(view.order());
        MemoryBuffer buffer(vjp_workspace_buffer_bytes<Int>(worker_count, dim, samples));
        auto buffer_alloc = buffer.template get_allocator<NumT, CACHE_LINE_ALIGNMENT>();
        std::vector<VJPWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        workspaces.emplace_back(dim, samples, dA.data(), alloc);
        for (size_t i = 1; i < worker_count; ++i) {
            NumT* local_dA = buffer_alloc.allocate(span_size);
            workspaces.emplace_back(dim, samples, local_dA, alloc);
            workspaces.back().dA.zero();
        }

        GLIntegrator::invoke(view.order(), [&](size_t q, int ln){
            auto& ws = workspaces[ln];
            auto [w_q, x_q] = view[q];

            double shift = x_q - 1.0;
            double dscale = dt * w_q;

            MatrixT last_dOut = dOut[n - 1];
            ws.integrator.sum_vjp(ws.barY, last_dOut, dscale);

            for (size_t k = n; k > 2; --k) {
                SpanT P = fwd_data.template prefix<MatPolicyT>(q, k);

                ws.barY.sample_update_vjp(
                    ws.dA,
                    A,
                    P,
                    shift,
                    ws.temp
                );

                ws.integrator.prefix_vjp(ws.barY, dt);

                MatrixT direct_dOut = dOut[k - 2];
                ws.integrator.sum_vjp_add(ws.barY, direct_dOut, dscale);
            }

            SpanT P = fwd_data.template prefix<MatPolicyT>(q, 2);

            ws.barY.sample_update_vjp(
                ws.dA,
                A,
                P,
                shift,
                ws.temp
            );

            ws.integrator.prefix_vjp(ws.barY, dt);

            ws.dA.add(ws.barY);
        });

        for (size_t i = 1; i < worker_count; ++i) dA.add(workspaces[i].dA);
    }

    template <Integrator Int, class GLIntegrator = GL_forloop>
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
        using AllocatorT = typename Int::allocator_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        using DynMatrixSpanT = DynMatrixSpan<NumT, MatPolicyT, AllocatorT>;
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

            none<Int, GLIntegrator>(
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

        size_t worker_count = GLIntegrator::lane_count(view.order());
        MemoryBuffer buffer(vjp_workspace_buffer_bytes<Int>(worker_count, dim, samples));
        auto buffer_alloc = buffer.template get_allocator<NumT, CACHE_LINE_ALIGNMENT>();
        std::vector<VJPWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        workspaces.emplace_back(dim, samples, dA.data(), alloc);
        for (size_t i = 1; i < worker_count; ++i) {
            NumT* local_dA = buffer_alloc.allocate(span_size);
            workspaces.emplace_back(dim, samples, local_dA, alloc);
            workspaces.back().dA.zero();
        }

        GLIntegrator::invoke(view.order(), [&](size_t q, int ln){
            auto& ws = workspaces[ln];
            auto [w_q, x_q] = view[q];

            double shift = x_q - 1.0;
            double dscale = dt * w_q;

            ws.integrator.sum_vjp(ws.barY, dOut, dscale);

            for (size_t k = n; k > 2; --k) {
                SpanT P = fwd_data.template prefix<MatPolicyT>(q, k);

                ws.barY.sample_update_vjp(
                    ws.dA,
                    A,
                    P,
                    shift,
                    ws.temp
                );

                ws.integrator.prefix_vjp(ws.barY, dt);

                ws.integrator.sum_vjp_add(ws.barY, dOut, dscale);
            }

            SpanT P = fwd_data.template prefix<MatPolicyT>(q, 2);

            ws.barY.sample_update_vjp(
                ws.dA,
                A,
                P,
                shift,
                ws.temp
            );

            ws.integrator.prefix_vjp(ws.barY, dt);

            ws.dA.add(ws.barY);
        });

        for (size_t i = 1; i < worker_count; ++i) dA.add(workspaces[i].dA);
    }


}


#endif
