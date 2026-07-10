#ifndef __LINALG_BLAS_HPP__
#define __LINALG_BLAS_HPP__
#include "generic.hpp"
#include <mkl.h>
#include <execution>

namespace Magnus {
    template <class NumT>
    void blas_matmul(size_t dim, const NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, NumT* MAGNUS_RESTRICT out ) {
        if constexpr ( std::is_same_v<NumT, float> ) {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dim, dim, dim,
                1,
                a,
                dim,
                b,
                dim,
                0.0f,
                out,
                dim
            );

        }
        else if constexpr ( std::is_same_v<NumT, double> ) {
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dim, dim, dim,
                1,
                a,
                dim,
                b,
                dim,
                0.0f,
                out,
                dim
            );

        }
        else if constexpr ( std::is_same_v<NumT, std::complex<float>> ) {
            std::complex<float> alpha(1, 0);
            std::complex<float> beta(0, 0);
            cblas_cgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dim, dim, dim,
                (const void*)&alpha,
                a,
                dim,
                b,
                dim,
                (const void*)&beta,
                out,
                dim
            );

        }
        else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
            std::complex<double> alpha(1, 0);
            std::complex<double> beta(0, 0);
            cblas_zgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dim, dim, dim,
                (const void*)&alpha,
                a,
                dim,
                b,
                dim,
                (const void*)&beta,
                out,
                dim
            );

        }
    }

    template <class NumT>
    void blas_matadd( size_t dim, NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, double scalar ) {
        size_t size = dim * dim;

        if constexpr ( std::is_same_v<NumT, float> ) {
            cblas_saxpy( size, scalar, b, 1, a, 1);
        }
        else if constexpr ( std::is_same_v<NumT, double> ) {
            cblas_daxpy( size, scalar, b, 1, a, 1);
        }
        else if constexpr ( std::is_same_v<NumT, std::complex<float>> ) {
            std::complex<float> alpha(scalar, 0);
            cblas_caxpy( size, (const void*)&alpha, b, 1, a, 1);
        }
        else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
            std::complex<double> alpha(scalar, 0);
            cblas_zaxpy( size, (const void*)&alpha, b, 1, a, 1);
        } else {
            for (size_t i = 0; i < size; ++i) a[i] += b[i] * scalar;
        }

    }

    template <class NumT>
    void blas_matscale(size_t dim, NumT* MAGNUS_RESTRICT a, double scalar) {
        size_t size = dim * dim;

        if constexpr ( std::is_same_v<NumT, float> ) {
            cblas_sscal( size, scalar, a, 1 );
        }
        else if constexpr ( std::is_same_v<NumT, double> ) {
            cblas_dscal( size, scalar, a, 1 );
        }
        else if constexpr ( std::is_same_v<NumT, std::complex<float>>) {
            cblas_csscal( size, scalar, a, 1 );
        }
        else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
            cblas_zdscal( size, scalar, a, 1 );
        }
        else {
            for (size_t i = 0; i < size; ++i) a[i] *= scalar;
        }
    }

    template <class NumT>
    void blas_matwcopy(size_t total, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst) {
        if constexpr ( std::is_same_v<NumT, float> ) {
            cblas_scopy( total, src, 1, dst, 1 );
        }
        else if constexpr ( std::is_same_v<NumT, double> ) {
            cblas_dcopy( total, src, 1, dst, 1 );
        }
        else if constexpr ( std::is_same_v<NumT, std::complex<float>>) {
            cblas_ccopy( total, src, 1, dst, 1 );
        }
        else if constexpr ( std::is_same_v<NumT, std::complex<double>> ) {
            cblas_zcopy( total, src, 1, dst, 1 );
        }
        else {
            for (size_t i = 0; i < total; ++i) dst[i] = src[i];
        }
    }

    template <class NumT>
    void blas_matcopy(size_t dim, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst) {
        blas_matwcopy<NumT>(dim * dim, src, dst);
    }

    template <class NumT>
    void blas_zero(size_t dim, NumT* MAGNUS_RESTRICT dst) {
        size_t size = dim * dim;
        static constexpr size_t PAR_THRESHOLD = 1 << 26;
        if ( size >= PAR_THRESHOLD ) {
            std::fill_n( std::execution::par, dst, size, NumT{0} );
            return;
        }
        std::fill_n(dst, size, NumT{0});
    }

    template <class NumT>
    using BlasPolicy = GenericMatrixPolicy<NumT, blas_matmul<NumT>, blas_matadd<NumT>, blas_matscale<NumT>, blas_matcopy<NumT>, blas_matwcopy<NumT>, blas_zero<NumT>>;

}

#endif