#ifndef __MAGNUS_HPP__
#define __MAGNUS_HPP__
#include "linalg/matrix.hpp"
#include "integration/integrate.hpp"
#include "util/gausslegendre.hpp"
#include "util/extra.hpp"
#include "graddata.hpp"
#include "mem/memorybuffer.hpp"

namespace Magnus {

    template <Integrator Int, VJP::DataRecorder<typename Int::matrix_span_t> VJPDataT, class GLIntegrator = GL_forloop>
    void one(
        typename Int::matrix_t& out,
        size_t n, 
        typename Int::matrix_span_t& A,
        double t0, double tf,
        VJPDataT& vjp_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using namespace Magnus;
        using NumT = typename Int::numeric_t;
        using AllocatorT = typename Int::allocator_t;
        using MatrixViewT = typename Int::matrix_t;
        using DynMatrixT = DynMatrix<NumT, typename Int::matrix_policy_t, AllocatorT>;
        using DynMatrixSpanT = DynMatrixSpan<NumT, typename Int::matrix_policy_t, AllocatorT>;

        // Bookkeeping stuff
        size_t mat_dim = out.dim();
        size_t mat_size = out.size();
        size_t sample_len = A.length();
        double dt = (tf - t0) / (sample_len - 1);

        out.zero();
        
        if ( n == 1 ) {
            Int integrator( mat_dim, alloc );
            integrator.sum( A, out, dt );
            return;
        }

        // GL weights/nodes
        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        size_t worker_count = GLIntegrator::lane_count(view.order());
        size_t local_buffer_bytes = fwd_workspace_buffer_bytes<Int>(
            worker_count,
            mat_dim,
            1
        );
        MemoryBuffer buf(local_buffer_bytes);
        auto local_buffer_alloc = buf.template get_allocator<NumT, CACHE_LINE_ALIGNMENT>();

        std::vector<FwdWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        workspaces.emplace_back(mat_dim, sample_len, 1, out.data(), alloc);
        for (size_t i = 1; i < worker_count; ++i) workspaces.emplace_back(mat_dim, sample_len, 1, local_buffer_alloc.allocate(mat_size), alloc );

        GLIntegrator::invoke(
            view.order(),
            [&](size_t q, int ln){
                auto [ w_q, x_q ] = view[q];
                auto& ws = workspaces[ln];
                double shift = x_q - 1;
                double final_scale = dt * w_q;

                ws.Y.copy_from(A);
                MatrixViewT temp = ws.integrator.borrow_scratch(); // maybe consider a refactor moving this elsewhere? view_t copy is cheap, so it should be ok

                for ( size_t k = 2; k <= n; ++k ) {
                    ws.integrator.prefix( ws.Y, dt );

                    vjp_data.record_prefix(q, k, ws.Y);

                    MatrixViewT total = ws.Y[ sample_len - 1 ];
                    ws.total_copy.copy_from(total);
                    ws.Y.sample_update(A, ws.total_copy, shift, temp);
                }

                MatrixViewT partial_out = ws.partial_out[0];
                ws.integrator.sum(ws.Y, partial_out, final_scale);
            }
        );

        for (size_t stride = 1; stride < worker_count; stride *= 2) {
            for (size_t i = 0; i + stride < worker_count; i += 2 * stride) {
                workspaces[i].partial_out.add(workspaces[i + stride].partial_out);
            }
        }

    }

    template <Integrator Int, VJP::DataRecorder<typename Int::matrix_span_t> VJPDataT, class GLIntegrator = GL_forloop>
    void many(
        typename Int::matrix_span_t& out,
        typename Int::matrix_span_t& A,
        double t0, double tf,
        VJPDataT& vjp_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using namespace Magnus;
        using NumT = typename Int::numeric_t;
        using AllocatorT = typename Int::allocator_t;
        using MatrixViewT = typename Int::matrix_t;
        using DynMatrixT = DynMatrix<NumT, typename Int::matrix_policy_t, AllocatorT>;
        using DynMatrixSpanT = DynMatrixSpan<NumT, typename Int::matrix_policy_t, AllocatorT>;

        // Bookkeeping stuff
        size_t n = out.length();
        size_t mat_dim = A.mat_dim();
        size_t mat_size = A.mat_size();
        size_t sample_len = A.length();
        double dt = (tf - t0) / (sample_len - 1);
        
        out.zero();
        
        if ( n == 1 ) {
            Int integrator( mat_dim, alloc );
            MatrixViewT first_term = out[0];
            integrator.sum( A, first_term, dt );
            return;
        }

        // GL weights/nodes
        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        size_t worker_count = GLIntegrator::lane_count(view.order());
        size_t local_buffer_bytes = fwd_workspace_buffer_bytes<Int>(
            worker_count,
            mat_dim,
            n
        );
        MemoryBuffer buf(local_buffer_bytes);
        auto local_buffer_alloc = buf.template get_allocator<NumT, CACHE_LINE_ALIGNMENT>();

        std::vector<FwdWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        workspaces.emplace_back(mat_dim, sample_len, n, out.data(), alloc);
        for (size_t i = 1; i < worker_count; ++i) workspaces.emplace_back(mat_dim, sample_len, n, local_buffer_alloc.allocate(mat_size * n), alloc );

        MatrixViewT first_term = workspaces[0].partial_out[0];
        workspaces[0].integrator.sum(A, first_term, dt);

        GLIntegrator::invoke(
            view.order(),
            [&](size_t q, int ln){
                auto [ w_q, x_q ] = view[q];
                auto& ws = workspaces[ln];
                double shift = x_q - 1;
                double final_scale = dt * w_q;

                ws.Y.copy_from(A);
                MatrixViewT temp = ws.integrator.borrow_scratch();

                for ( size_t k = 2; k <= n; ++k ) {
                    ws.integrator.prefix( ws.Y, dt );

                    vjp_data.record_prefix(q, k, ws.Y);

                    MatrixViewT total = ws.Y[ sample_len - 1 ];
                    ws.total_copy.copy_from(total);
                    ws.Y.sample_update(A, ws.total_copy, shift, temp);

                    MatrixViewT term = ws.partial_out[k - 1];
                    ws.integrator.sum(ws.Y, term, final_scale);
                }
            }
        );

        for (size_t stride = 1; stride < worker_count; stride *= 2) {
            for (size_t i = 0; i + stride < worker_count; i += 2 * stride) {
                workspaces[i].partial_out.add(workspaces[i + stride].partial_out);
            }
        }
    }

    template <Integrator Int, VJP::DataRecorder<typename Int::matrix_span_t> VJPDataT, class GLIntegrator = GL_forloop>
    void sum(
        typename Int::matrix_t& out,
        size_t n, 
        typename Int::matrix_span_t& A,
        double t0, double tf,
        VJPDataT& vjp_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using namespace Magnus;
        using NumT = typename Int::numeric_t;
        using AllocatorT = typename Int::allocator_t;
        using MatrixViewT = typename Int::matrix_t;
        using DynMatrixT = DynMatrix<NumT, typename Int::matrix_policy_t, AllocatorT>;
        using DynMatrixSpanT = DynMatrixSpan<NumT, typename Int::matrix_policy_t, AllocatorT>;

        // Bookkeeping stuff
        size_t mat_dim = out.dim();
        size_t mat_size = out.size();
        size_t sample_len = A.length();
        double dt = (tf - t0) / (sample_len - 1);

        out.zero();

        if ( n == 1 ) {
            Int integrator( mat_dim, alloc );
            integrator.sum( A, out, dt );
            return;
        }

        // GL weights/nodes
        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        size_t worker_count = GLIntegrator::lane_count(view.order());
        size_t local_buffer_bytes = fwd_workspace_buffer_bytes<Int>(
            worker_count,
            mat_dim,
            1
        );
        MemoryBuffer buf(local_buffer_bytes);
        auto local_buffer_alloc = buf.template get_allocator<NumT, CACHE_LINE_ALIGNMENT>();

        std::vector<FwdWorkspace<Int>> workspaces;
        workspaces.reserve(worker_count);
        workspaces.emplace_back(mat_dim, sample_len, 1, out.data(), alloc);
        for (size_t i = 1; i < worker_count; ++i) workspaces.emplace_back(mat_dim, sample_len, 1, local_buffer_alloc.allocate(mat_size), alloc );

        MatrixViewT partial_out = workspaces[0].partial_out[0];
        workspaces[0].integrator.sum(A, partial_out, dt);

        GLIntegrator::invoke(
            view.order(),
            [&](size_t q, int ln){
                auto [ w_q, x_q ] = view[q];
                auto& ws = workspaces[ln];
                double shift = x_q - 1;
                double final_scale = dt * w_q;

                ws.Y.copy_from(A);
                MatrixViewT temp = ws.integrator.borrow_scratch();

                for ( size_t k = 2; k <= n; ++k ) {
                    ws.integrator.prefix( ws.Y, dt );

                    vjp_data.record_prefix(q, k, ws.Y);

                    MatrixViewT total = ws.Y[ sample_len - 1 ];
                    ws.total_copy.copy_from(total);
                    ws.Y.sample_update(A, ws.total_copy, shift, temp);

                    MatrixViewT partial_out = ws.partial_out[0];
                    ws.integrator.sum(ws.Y, partial_out, final_scale);
                }
            }
        );

        for (size_t stride = 1; stride < worker_count; stride *= 2) {
            for (size_t i = 0; i + stride < worker_count; i += 2 * stride) {
                workspaces[i].partial_out.add(workspaces[i + stride].partial_out);
            }
        }
    }

}

#endif
