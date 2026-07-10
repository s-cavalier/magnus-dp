#ifndef __LINALG_FIXED_HPP__
#define __LINALG_FIXED_HPP__
#include "generic.hpp"

namespace Magnus {

    template <class NumT, size_t Dim>
    void fixed_dim_matmul(
        [[maybe_unused]] size_t dim,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        NumT* MAGNUS_RESTRICT out
    ) {
        if constexpr ( Dim == 2 ) {
            out[0] = a[0] * b[0] + a[1] * b[2];
            out[1] = a[0] * b[1] + a[1] * b[3];
            out[2] = a[2] * b[0] + a[3] * b[2];
            out[3] = a[2] * b[1] + a[3] * b[3];
            return;
        }

        for (size_t i = 0; i < Dim; ++i) {
            for (size_t j = 0; j < Dim; ++j) {
                out[i * Dim + j] = 0;

                for (size_t k = 0; k < Dim; ++k) out[i * Dim + j] += a[i * Dim + k] * b[k * Dim + j];
            }
        }

    }

    template <class NumT, size_t Dim>
    void fixed_dim_matadd(
        [[maybe_unused]] size_t dim,
        NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        double scalar
    ) {
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < Dim * Dim; ++i) a[i] += b[i] * x;
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matscale( [[maybe_unused]] size_t dim, NumT* MAGNUS_RESTRICT a, double scalar ) {
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < Dim * Dim; ++i) a[i] *= x;
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matwcopy( size_t total, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
        for ( size_t i = 0; i < total; ++i ) dst[i] = src[i];
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matcopy( [[maybe_unused]] size_t, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
        for ( size_t i = 0; i < Dim * Dim; ++i ) dst[i] = src[i];
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matzero( [[maybe_unused]] size_t, NumT* MAGNUS_RESTRICT dst ) {
        for (size_t i = 0; i < Dim * Dim; ++i) dst[i] = 0;
    }

    template <class NumT, size_t Dim>
    void fixed_dim_sample_update(
        [[maybe_unused]] size_t dim,
        size_t len,
        const NumT* MAGNUS_RESTRICT A,
        NumT* MAGNUS_RESTRICT Y,
        const NumT* MAGNUS_RESTRICT total,
        double shift,
        NumT* MAGNUS_RESTRICT temp
    ) {
        if constexpr ( Dim == 2 ) {
            auto x = scalar_as_num<NumT>(shift);

            for (size_t i = 0; i < len; ++i) {
                const NumT* MAGNUS_RESTRICT a = A + i * 4;
                NumT* MAGNUS_RESTRICT y = Y + i * 4;

                NumT b0 = y[0] + total[0] * x;
                NumT b1 = y[1] + total[1] * x;
                NumT b2 = y[2] + total[2] * x;
                NumT b3 = y[3] + total[3] * x;

                y[0] = a[0] * b0 + a[1] * b2;
                y[1] = a[0] * b1 + a[1] * b3;
                y[2] = a[2] * b0 + a[3] * b2;
                y[3] = a[2] * b1 + a[3] * b3;
            }
        }
        else {
            generic_sample_update<
                NumT,
                fixed_dim_matmul<NumT, Dim>,
                fixed_dim_matadd<NumT, Dim>,
                fixed_dim_matcopy<NumT, Dim>
            >(dim, len, A, Y, total, shift, temp);
        }
    }

    template <class NumT, size_t Dim>
    using FixedDimPolicy = GenericMatrixPolicy<NumT, fixed_dim_matmul<NumT, Dim>, fixed_dim_matadd<NumT, Dim>, fixed_dim_matscale<NumT, Dim>, fixed_dim_matcopy<NumT, Dim>, fixed_dim_matwcopy<NumT, Dim>, fixed_dim_matzero<NumT, Dim>, fixed_dim_sample_update<NumT, Dim>>;


}

#endif