#ifndef __MAGNUS_HPP__
#define __MAGNUS_HPP__
#include "matrix.hpp"
#include "integrate.hpp"
#include <vector>

template <class NumT, class Allocator = std::allocator<NumT>>
void magnus(
    Magnus::MatrixView<NumT>& out,
    size_t n, 
    NumT* data, // data has mat_dim * mat_dim * sample_len size, where mat_dim is based on the out matrix view
    NumT t0, NumT tf,
    size_t sample_len,
    const Allocator& alloc = Allocator()
) {
    using namespace Magnus;

    // Bookkeeping stuff
    size_t mat_dim = out.dim();
    size_t matrix_size = mat_dim * mat_dim;
    size_t total_data_size = matrix_size * sample_len;
    NumT dt = (tf - t0) / sample_len;
    auto A = [=](size_t i) { return MatrixView<NumT>(data + i * matrix_size, mat_dim); };    
    out.fill( []() constexpr { return 0; } );

    if ( n == 1 ) {
        for (size_t i = 0; i < sample_len; ++i) out.add(A(i));
        out.scale(dt);
        return;
    }

    // GL weights/nodes
    GLTable::DataView view = GLTable::get()->get_order(n);

    // DP buffer space
    std::vector<NumT, Allocator> Y_storage(alloc);
    Y_storage.resize( total_data_size );
    NumT* Y_data = Y_storage.data();
    auto Y = [=](size_t i){ return MatrixView<NumT>(Y_data + i * matrix_size, mat_dim); };

    DynMatrix<NumT, Allocator> temp( mat_dim, alloc );

    for ( size_t q = 0; q < view.order(); ++q ) {
        auto [ w_q, x_q ] = view[q];
        NumT shift = x_q - 1;
        NumT final_scale = dt * w_q;

        std::copy_n( data, total_data_size, Y_data );
        
        for ( size_t k = 2; k <= n; ++k ) {
            Y(0).scale(dt);

            for (size_t i = 1; i < sample_len; ++i) Y(i).scale(dt).add( Y(i - 1) );

            MatrixView<NumT> total = Y( sample_len - 1 );

            for (size_t i = 0; i < sample_len; ++i) {
                MatrixView<NumT> Y_i = Y(i);
                Y_i.add_scaled( total, shift );
                matmul( A(i), Y_i, temp );
                Y_i.copy_from(temp);
            };
            
        }   

        for (size_t i = 0; i < sample_len; ++i) out.add_scaled(Y(i), final_scale);
    }

}



#endif