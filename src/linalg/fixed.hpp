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
    void fixed_dim_matmul_vjp(
        [[maybe_unused]] size_t dim,
        NumT* MAGNUS_RESTRICT da,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT dout,
        const NumT* b,
        NumT* db
    ) {
        if constexpr (Dim == 2) {
            NumT b0 = b[0];
            NumT b1 = b[1];
            NumT b2 = b[2];
            NumT b3 = b[3];

            da[0] += dout[0] * b0 + dout[1] * b1;
            da[1] += dout[0] * b2 + dout[1] * b3;
            da[2] += dout[2] * b0 + dout[3] * b1;
            da[3] += dout[2] * b2 + dout[3] * b3;

            db[0] = a[0] * dout[0] + a[2] * dout[2];
            db[1] = a[0] * dout[1] + a[2] * dout[3];
            db[2] = a[1] * dout[0] + a[3] * dout[2];
            db[3] = a[1] * dout[1] + a[3] * dout[3];
            return;
        }

        for (size_t i = 0; i < Dim; ++i) {
            for (size_t k = 0; k < Dim; ++k) {
                NumT value = NumT{0};
                for (size_t j = 0; j < Dim; ++j) value += dout[i * Dim + j] * b[k * Dim + j];
                da[i * Dim + k] += value;
            }
        }

        for (size_t k = 0; k < Dim; ++k) {
            for (size_t j = 0; j < Dim; ++j) {
                NumT value = NumT{0};
                for (size_t i = 0; i < Dim; ++i) value += a[i * Dim + k] * dout[i * Dim + j];
                db[k * Dim + j] = value;
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
    void fixed_dim_matwzero( size_t total, NumT* MAGNUS_RESTRICT dst ) {
        for (size_t i = 0; i < total; ++i) dst[i] = NumT{0};
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matwadd( size_t total, NumT* MAGNUS_RESTRICT dst, const NumT* MAGNUS_RESTRICT src ) {
        for (size_t i = 0; i < total; ++i) dst[i] += src[i];
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
    MAGNUS_ALWAYS_INLINE void fixed_dim_sample_update(
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
    void fixed_dim_sample_update_vjp(
        size_t dim,
        size_t len,
        NumT* MAGNUS_RESTRICT dA,
        const NumT* MAGNUS_RESTRICT A,
        NumT* MAGNUS_RESTRICT barY,
        const NumT* MAGNUS_RESTRICT prefix,
        double shift,
        NumT* MAGNUS_RESTRICT temp
    ) {
        if constexpr (Dim == 2) {
            auto x = scalar_as_num<NumT>(shift);
            auto one_plus_x = scalar_as_num<NumT>(1.0 + shift);
            size_t last = len - 1;
            const NumT* MAGNUS_RESTRICT total = prefix + last * 4;

            auto reverse_sample = [&](size_t sample) {
                NumT* MAGNUS_RESTRICT da = dA + sample * 4;
                const NumT* MAGNUS_RESTRICT a = A + sample * 4;
                const NumT* MAGNUS_RESTRICT g = barY + sample * 4;
                const NumT* MAGNUS_RESTRICT p = prefix + sample * 4;

                NumT b0 = p[0] + total[0] * x;
                NumT b1 = p[1] + total[1] * x;
                NumT b2 = p[2] + total[2] * x;
                NumT b3 = p[3] + total[3] * x;
                NumT g0 = g[0];
                NumT g1 = g[1];
                NumT g2 = g[2];
                NumT g3 = g[3];

                da[0] += g0 * b0 + g1 * b1;
                da[1] += g0 * b2 + g1 * b3;
                da[2] += g2 * b0 + g3 * b1;
                da[3] += g2 * b2 + g3 * b3;

                temp[0] = a[0] * g0 + a[2] * g2;
                temp[1] = a[0] * g1 + a[2] * g3;
                temp[2] = a[1] * g0 + a[3] * g2;
                temp[3] = a[1] * g1 + a[3] * g3;
            };

            reverse_sample(last);
            NumT* MAGNUS_RESTRICT bar_total = barY + last * 4;
            bar_total[0] = temp[0] * one_plus_x;
            bar_total[1] = temp[1] * one_plus_x;
            bar_total[2] = temp[2] * one_plus_x;
            bar_total[3] = temp[3] * one_plus_x;

            for (size_t sample = last; sample-- > 0;) {
                reverse_sample(sample);
                NumT* MAGNUS_RESTRICT g = barY + sample * 4;
                bar_total[0] += temp[0] * x;
                bar_total[1] += temp[1] * x;
                bar_total[2] += temp[2] * x;
                bar_total[3] += temp[3] * x;
                g[0] = temp[0];
                g[1] = temp[1];
                g[2] = temp[2];
                g[3] = temp[3];
            }
        } else {
            generic_sample_update_vjp<
                NumT,
                fixed_dim_matmul_vjp<NumT, Dim>,
                fixed_dim_matadd<NumT, Dim>,
                fixed_dim_matcopy<NumT, Dim>
            >(dim, len, dA, A, barY, prefix, shift, temp);
        }
    }

    template <class NumT, size_t Dim>
    using FixedDimPolicy = GenericMatrixPolicy<NumT, fixed_dim_matmul<NumT, Dim>, fixed_dim_matmul_vjp<NumT, Dim>, fixed_dim_matadd<NumT, Dim>, fixed_dim_matscale<NumT, Dim>, fixed_dim_matcopy<NumT, Dim>, fixed_dim_matwcopy<NumT, Dim>, fixed_dim_matzero<NumT, Dim>, fixed_dim_matwzero<NumT, Dim>, fixed_dim_matwadd<NumT, Dim>, fixed_dim_sample_update<NumT, Dim>, fixed_dim_sample_update_vjp<NumT, Dim>>;


}

#endif
