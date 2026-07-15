#ifndef __MAGNUS_HPP__
#define __MAGNUS_HPP__
#include "linalg/matrix.hpp"
#include "integration/integrate.hpp"
#include "util/gausslegendre.hpp"
#include "util/extra.hpp"
#include "graddata.hpp"

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
        using MatrixSpanT = typename Int::matrix_span_t;

        // Bookkeeping stuff
        size_t mat_dim = out.dim();
        size_t matrix_size = mat_dim * mat_dim;
        size_t sample_len = A.length();
        size_t total_data_size = matrix_size * sample_len;
        double dt = (tf - t0) / (sample_len - 1);

        out.zero();
        
        Int integrator( mat_dim, alloc );

        if ( n == 1 ) {
            integrator.sum( A, out, dt );
            return;
        }

        // GL weights/nodes
        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        // DP buffer space
        auto Y_data = allocate_unique<NumT>(total_data_size, alloc);
        auto total_data = allocate_unique<NumT>(matrix_size, alloc);
        MatrixSpanT Y( Y_data.get(), mat_dim, sample_len );
        MatrixViewT total_copy( total_data.get(), mat_dim );

        MatrixViewT temp = integrator.borrow_scratch();

        GLIntegrator::invoke(view.order(), [&](size_t q){
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;
            double final_scale = dt * w_q;

            Y.copy_from(A);
            
            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                vjp_data.record_prefix(q, k, Y);
                
                MatrixViewT total = Y[ sample_len - 1 ];
                total_copy.copy_from(total);
                Y.sample_update(A, total_copy, shift, temp);
            }   

            integrator.sum(Y, out, final_scale);
        });

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
        using MatrixSpanT = typename Int::matrix_span_t;

        // Bookkeeping stuff
        size_t n = out.length();
        size_t mat_dim = A.mat_dim();
        size_t matrix_size = mat_dim * mat_dim;
        size_t sample_len = A.length();
        size_t total_data_size = matrix_size * sample_len;
        double dt = (tf - t0) / (sample_len - 1);
        
        std::fill_n( out.data(), out.size(), NumT(0) );
        
        Int integrator( mat_dim, alloc );

        MatrixViewT first_term = out[0];
        integrator.sum( A, first_term, dt );
        if ( n == 1 ) return;

        // GL weights/nodes
        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        // DP buffer space
        auto Y_data = allocate_unique<NumT>(total_data_size, alloc);
        auto total_data = allocate_unique<NumT>(matrix_size, alloc);
        MatrixSpanT Y( Y_data.get(), mat_dim, sample_len );
        MatrixViewT total_copy( total_data.get(), mat_dim );

        MatrixViewT temp = integrator.borrow_scratch();

        GLIntegrator::invoke(view.order(), [&](size_t q){
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;
            double final_scale = dt * w_q;

            Y.copy_from(A);
            
            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                vjp_data.record_prefix(q, k, Y);

                MatrixViewT total = Y[ sample_len - 1 ];
                total_copy.copy_from(total);
                Y.sample_update(A, total_copy, shift, temp);
                
                MatrixViewT term = out[k - 1];
                integrator.sum(Y, term, final_scale);
            }   
        });
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
        using MatrixSpanT = typename Int::matrix_span_t;

        // Bookkeeping stuff
        size_t mat_dim = out.dim();
        size_t matrix_size = mat_dim * mat_dim;
        size_t sample_len = A.length();
        size_t total_data_size = matrix_size * sample_len;
        double dt = (tf - t0) / (sample_len - 1);

        out.zero();

        Int integrator( mat_dim, alloc );

        integrator.sum( A, out, dt );
        if ( n == 1 ) return;

        // GL weights/nodes
        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order( (n + 3) / 2 );

        // DP buffer space
        auto Y_data = allocate_unique<NumT>(total_data_size, alloc);
        auto total_data = allocate_unique<NumT>(matrix_size, alloc);
        MatrixSpanT Y( Y_data.get(), mat_dim, sample_len );
        MatrixViewT total_copy( total_data.get(), mat_dim );

        MatrixViewT temp = integrator.borrow_scratch();

        GLIntegrator::invoke(view.order(), [&](size_t q){
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;
            double final_scale = dt * w_q;

            Y.copy_from(A);
            
            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                vjp_data.record_prefix(q, k, Y);

                MatrixViewT total = Y[ sample_len - 1 ];
                total_copy.copy_from(total);
                Y.sample_update(A, total_copy, shift, temp);

                integrator.sum(Y, out, final_scale);
            }   
        });
    }

}

#endif
