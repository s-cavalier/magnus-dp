#ifndef __MAGNUS_HPP__
#define __MAGNUS_HPP__
#include "matrix.hpp"
#include "integrate.hpp"
#include "gausslegendre.hpp"

namespace Magnus {

    template <Integrator Int>
    void one(
        typename Int::matrix_t& out,
        size_t n, 
        typename Int::matrix_span_t& A,
        double t0, double tf,
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
        typename Int::allocator_t usable_alloc = alloc;
        NumT* Y_data = usable_alloc.allocate(total_data_size);
        MatrixSpanT Y( Y_data, mat_dim, sample_len );

        MatrixViewT temp = integrator.borrow_scratch();

        for ( size_t q = 0; q < view.order(); ++q ) {
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;
            double final_scale = dt * w_q;

            Y.copy_from(A);
            
            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                MatrixViewT total = Y[ sample_len - 1 ];

                for (size_t i = 0; i < sample_len; ++i) {
                    MatrixViewT Y_i = Y[i];
                    Y_i.add( total, shift );
                    temp.save_matmul( A[i], Y_i );
                    Y_i.copy_from(temp);
                };
                
            }   

            integrator.sum(Y, out, final_scale);
        }

        usable_alloc.deallocate(Y_data, total_data_size);
    }

    template <Integrator Int>
    void many(
        typename Int::matrix_span_t& out,
        typename Int::matrix_span_t& A,
        double t0, double tf,
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
        typename Int::allocator_t usable_alloc = alloc;
        NumT* Y_data = usable_alloc.allocate(total_data_size);
        MatrixSpanT Y( Y_data, mat_dim, sample_len );

        MatrixViewT temp = integrator.borrow_scratch();

        for ( size_t q = 0; q < view.order(); ++q ) {
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;
            double final_scale = dt * w_q;

            Y.copy_from(A);
            
            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                MatrixViewT total = Y[ sample_len - 1 ];

                for (size_t i = 0; i < sample_len; ++i) {
                    MatrixViewT Y_i = Y[i];
                    Y_i.add(total, shift);
                    temp.save_matmul( A[i], Y_i );
                    Y_i.copy_from(temp);
                };
                
                MatrixViewT term = out[k - 1];
                integrator.sum(Y, term, final_scale);
            }   
        }

        usable_alloc.deallocate(Y_data, total_data_size);

    }

    template <Integrator Int>
    void sum(
        typename Int::matrix_t& out,
        size_t n, 
        typename Int::matrix_span_t& A,
        double t0, double tf,
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
        typename Int::allocator_t usable_alloc = alloc;
        NumT* Y_data = usable_alloc.allocate(total_data_size);
        MatrixSpanT Y( Y_data, mat_dim, sample_len );

        MatrixViewT temp = integrator.borrow_scratch();

        for ( size_t q = 0; q < view.order(); ++q ) {
            auto [ w_q, x_q ] = view[q];
            double shift = x_q - 1;
            double final_scale = dt * w_q;

            Y.copy_from(A);
            
            for ( size_t k = 2; k <= n; ++k ) {
                integrator.prefix( Y, dt );

                MatrixViewT total = Y[ sample_len - 1 ];

                for (size_t i = 0; i < sample_len; ++i) {
                    MatrixViewT Y_i = Y[i];
                    Y_i.add(total, shift);
                    temp.save_matmul( A[i], Y_i );
                    Y_i.copy_from(temp);
                };

                integrator.sum(Y, out, final_scale);
            }   
        }

        usable_alloc.deallocate(Y_data, total_data_size);
    }

}

#endif
