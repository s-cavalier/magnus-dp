#ifndef __LINALG_MANUAL_HPP__
#define __LINALG_MANUAL_HPP__
#include "generic.hpp"

namespace Magnus {

    template <class NumT>
    void manual_matmul( size_t dim, const NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, NumT* MAGNUS_RESTRICT out ) {
        if (dim == 0) return;

        for (size_t i = 0; i < dim; ++i) {
            NumT* MAGNUS_RESTRICT out_row = out + i * dim;
            const NumT a_i0 = a[i * dim];

            for (size_t j = 0; j < dim; ++j) out_row[j] = a_i0 * b[j];

            for (size_t k = 1; k < dim; ++k) {
                const NumT a_ik = a[i * dim + k];
                const NumT* MAGNUS_RESTRICT b_row = b + k * dim;

                for (size_t j = 0; j < dim; ++j) out_row[j] += a_ik * b_row[j];
            }
        }
    }

    template <class NumT>
    void manual_matmul_vjp(
        size_t dim,
        NumT* MAGNUS_RESTRICT da,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT dout,
        const NumT* b,
        NumT* db
    ) {
        for (size_t i = 0; i < dim; ++i) {
            for (size_t k = 0; k < dim; ++k) {
                NumT value = NumT{0};
                for (size_t j = 0; j < dim; ++j) value += dout[i * dim + j] * b[k * dim + j];
                da[i * dim + k] += value;
            }
        }

        if (dim == 0) return;

        for (size_t k = 0; k < dim; ++k) {
            NumT* MAGNUS_RESTRICT db_row = db + k * dim;
            const NumT a_0k = a[k];

            for (size_t j = 0; j < dim; ++j) db_row[j] = a_0k * dout[j];

            for (size_t i = 1; i < dim; ++i) {
                const NumT a_ik = a[i * dim + k];
                const NumT* MAGNUS_RESTRICT dout_row = dout + i * dim;

                for (size_t j = 0; j < dim; ++j) db_row[j] += a_ik * dout_row[j];
            }
        }
    }

    template <class NumT>
    void manual_matadd(size_t dim, NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, double scalar) {
        size_t size = dim * dim;
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < size; ++i) a[i] += b[i] * x;
    }

    template <class NumT>
    void manual_matscale(size_t dim, NumT* MAGNUS_RESTRICT a, double scalar) {
        size_t size = dim * dim;
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < size; ++i) a[i] *= x;
    }

    template <class NumT>
    void manual_matwcopy(size_t total, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst) {
        std::copy_n(src, total, dst);
    }

    template <class NumT>
    void manual_matwzero(size_t total, NumT* MAGNUS_RESTRICT dst) {
        std::fill_n(dst, total, NumT{0});
    }

    template <class NumT>
    void manual_matwadd(size_t total, NumT* MAGNUS_RESTRICT dst, const NumT* MAGNUS_RESTRICT src) {
        for (size_t i = 0; i < total; ++i) dst[i] += src[i];
    }

    template <class NumT>
    void manual_matcopy(size_t dim, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst) {
        manual_matwcopy(dim * dim, src, dst);
    }

    template <class NumT>
    void manual_matzero(size_t dim, NumT* MAGNUS_RESTRICT dst) {
        std::fill_n(dst, dim * dim, NumT{0});
    }

    template <class NumT>
    using ManualPolicy = GenericMatrixPolicy<NumT, manual_matmul<NumT>, manual_matmul_vjp<NumT>, manual_matadd<NumT>, manual_matscale<NumT>, manual_matcopy<NumT>, manual_matwcopy<NumT>, manual_matzero<NumT>, manual_matwzero<NumT>, manual_matwadd<NumT>>;

}

#endif
