#ifndef __MAGNUS_HPP__
#define __MAGNUS_HPP__
#include <span>
#include "matrix.hpp"

template <class NumT>
Magnus::MatrixView<NumT> magnus(
    Magnus::MatrixView<NumT>& out,
    size_t n, 
    NumT* data, size_t mat_dim, 
    NumT t0, NumT tf,
    size_t sample_len
) {
    NumT dt = (tf - t0) / sample_len;
    auto t = [=](size_t i){ return t0 + i * dt; }

    

    for ( size_t i = 0; i <  )

}



#endif