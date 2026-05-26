#ifndef __MAGNUS_HPP__
#define __MAGNUS_HPP__
#include "matrix.hpp"
#include "integrate.hpp"
#include <vector>

template <class T>
concept Policy = requires ( Magnus::MatrixView<typename T::numeric_t>& m, const Magnus::MatrixView<typename T::numeric_t>& cm ) {
    typename T::numeric_t;
    typename T::allocator_t;
    typename T::integrator_t;
    requires Integrator<typename T::integrator_t>;
    requires std::same_as< typename T::integrator_t::numeric_t, typename T::numeric_t >;
    requires std::same_as< typename T::integrator_t::allocator_t, typename T::allocator_t >;

    { T::matmul_kernel( cm, cm, m ) } -> std::same_as<void>;
};

template <class NumT>
using MatMulKernelT = void(&)(const Magnus::MatrixView<NumT>&, const Magnus::MatrixView<NumT>&, Magnus::MatrixView<NumT>&);

template <
    class NumT, 
    Integrator Int, 
    MatMulKernelT<NumT> kernel = Magnus::dim2_matmul,
    class AllocatorT = std::allocator<NumT>
>
struct GenericPolicy {
    using numeric_t = NumT;
    using allocator_t = AllocatorT;
    using integrator_t = Int;
    
    static void matmul_kernel( const Magnus::MatrixView<NumT>& a, const Magnus::MatrixView<NumT>& b, Magnus::MatrixView<NumT>& out  ) {
        kernel(a, b, out);
    }
    
};

template <class NumT>
using StandardPolicy = GenericPolicy<NumT, BooleIntegrator<NumT>>;

template <Policy PolicyT>
void magnus_one(
    Magnus::MatrixView<typename PolicyT::numeric_t>& out,
    size_t n, 
    Magnus::MatrixSpan<typename PolicyT::numeric_t>& A,
    double t0, double tf,
    const typename PolicyT::allocator_t& alloc = typename PolicyT::allocator_t()
) {
    using namespace Magnus;
    using NumT = typename PolicyT::numeric_t;
    using AllocatorT = typename PolicyT::allocator_t;

    // Bookkeeping stuff
    size_t mat_dim = out.dim();
    size_t matrix_size = mat_dim * mat_dim;
    size_t sample_len = A.length();
    size_t total_data_size = matrix_size * sample_len;
    double dt = (tf - t0) / (sample_len - 1);
    
    out.zero();
    
    typename PolicyT::integrator_t integrator( mat_dim, alloc );

    if ( n == 1 ) {
        integrator.sum( A, out, dt );
        return;
    }

    // GL weights/nodes
    GLTable::DataView view = GLTable::get()->get_order( (n + 3) / 2 );

    // DP buffer space
    std::vector<NumT, AllocatorT> Y_storage(alloc);
    Y_storage.resize( total_data_size );
    NumT* Y_data = Y_storage.data();
    MatrixSpan<NumT> Y( Y_data, mat_dim, sample_len );

    MatrixView<NumT> temp = integrator.borrow_scratch();

    for ( size_t q = 0; q < view.order(); ++q ) {
        auto [ w_q, x_q ] = view[q];
        double shift = x_q - 1;
        double final_scale = dt * w_q;

        Y.copy_from(A);
        
        for ( size_t k = 2; k <= n; ++k ) {
            integrator.prefix( Y, dt );

            MatrixView<NumT> total = Y[ sample_len - 1 ];

            for (size_t i = 0; i < sample_len; ++i) {
                MatrixView<NumT> Y_i = Y[i];
                Y_i.add_scaled( total, shift );
                PolicyT::matmul_kernel( A[i], Y_i, temp );
                Y_i.copy_from(temp);
            };
            
        }   

        integrator.sum(Y, out, final_scale);
    }

}

template <Policy PolicyT>
void magnus_many(
    Magnus::MatrixSpan<typename PolicyT::numeric_t>& out,
    Magnus::MatrixSpan<typename PolicyT::numeric_t>& A,
    double t0, double tf,
    const typename PolicyT::allocator_t& alloc = typename PolicyT::allocator_t()
) {
    using namespace Magnus;
    using NumT = typename PolicyT::numeric_t;
    using AllocatorT = typename PolicyT::allocator_t;

    // Bookkeeping stuff
    size_t n = out.length();
    size_t mat_dim = A.mat_dim();
    size_t matrix_size = mat_dim * mat_dim;
    size_t sample_len = A.length();
    size_t total_data_size = matrix_size * sample_len;
    double dt = (tf - t0) / (sample_len - 1);
    
    std::fill_n( out.data(), out.size(), NumT(0) );
    
    typename PolicyT::integrator_t integrator( mat_dim, alloc );

    MatrixView<NumT> first_term = out[0];
    integrator.sum( A, first_term, dt );
    if ( n == 1 ) return;

    // GL weights/nodes
    GLTable::DataView view = GLTable::get()->get_order( (n + 3) / 2 );

    // DP buffer space
    std::vector<NumT, AllocatorT> Y_storage(alloc);
    Y_storage.resize( total_data_size );
    NumT* Y_data = Y_storage.data();
    MatrixSpan<NumT> Y( Y_data, mat_dim, sample_len );

    MatrixView<NumT> temp = integrator.borrow_scratch();

    for ( size_t q = 0; q < view.order(); ++q ) {
        auto [ w_q, x_q ] = view[q];
        double shift = x_q - 1;
        double final_scale = dt * w_q;

        Y.copy_from(A);
        
        for ( size_t k = 2; k <= n; ++k ) {
            integrator.prefix( Y, dt );

            MatrixView<NumT> total = Y[ sample_len - 1 ];

            for (size_t i = 0; i < sample_len; ++i) {
                MatrixView<NumT> Y_i = Y[i];
                Y_i.add_scaled( total, shift );
                PolicyT::matmul_kernel( A[i], Y_i, temp );
                Y_i.copy_from(temp);
            };
            
            MatrixView<NumT> term = out[k - 1];
            integrator.sum(Y, term, final_scale);
        }   
    }

}

template <Policy PolicyT>
void magnus_sum(
    Magnus::MatrixView<typename PolicyT::numeric_t>& out,
    size_t n, 
    Magnus::MatrixSpan<typename PolicyT::numeric_t>& A,
    double t0, double tf,
    const typename PolicyT::allocator_t& alloc = typename PolicyT::allocator_t()
) {
    using namespace Magnus;
    using NumT = typename PolicyT::numeric_t;
    using AllocatorT = typename PolicyT::allocator_t;

    // Bookkeeping stuff
    size_t mat_dim = out.dim();
    size_t matrix_size = mat_dim * mat_dim;
    size_t sample_len = A.length();
    size_t total_data_size = matrix_size * sample_len;
    double dt = (tf - t0) / (sample_len - 1);

    out.zero();

    typename PolicyT::integrator_t integrator( mat_dim, alloc );

    integrator.sum( A, out, dt );
    if ( n == 1 ) return;

    // GL weights/nodes
    GLTable::DataView view = GLTable::get()->get_order( (n + 3) / 2 );

    // DP buffer space
    std::vector<NumT, AllocatorT> Y_storage(alloc);
    Y_storage.resize( total_data_size );
    NumT* Y_data = Y_storage.data();
    MatrixSpan<NumT> Y( Y_data, mat_dim, sample_len );

    MatrixView<NumT> temp = integrator.borrow_scratch();

    for ( size_t q = 0; q < view.order(); ++q ) {
        auto [ w_q, x_q ] = view[q];
        double shift = x_q - 1;
        double final_scale = dt * w_q;

        Y.copy_from(A);
        
        for ( size_t k = 2; k <= n; ++k ) {
            integrator.prefix( Y, dt );

            MatrixView<NumT> total = Y[ sample_len - 1 ];

            for (size_t i = 0; i < sample_len; ++i) {
                MatrixView<NumT> Y_i = Y[i];
                Y_i.add_scaled( total, shift );
                PolicyT::matmul_kernel( A[i], Y_i, temp );
                Y_i.copy_from(temp);
            };

            integrator.sum(Y, out, final_scale);
        }   
    }

}

#endif
