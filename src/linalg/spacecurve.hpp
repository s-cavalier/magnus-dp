#ifndef __LINALG_SPACECURVE_HPP__
#define __LINALG_SPACECURVE_HPP__
#include "generic.hpp"

namespace Magnus::SpaceCurve {

    template <class NumT>
    void matmul(
        [[maybe_unused]] size_t,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        NumT* MAGNUS_RESTRICT c
    ) {
        c[0] = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        c[1] = b[0] * a[1] + a[2] * b[3] - a[3] * b[2];
        c[2] = b[0] * a[2] + a[3] * b[1] - a[1] * b[3];
        c[3] = b[0] * a[3] + a[1] * b[2] - a[2] * b[1];
    }

    template <class NumT>
    void matmul_vjp(
        [[maybe_unused]] size_t dim,
        NumT* MAGNUS_RESTRICT da,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT dout,
        const NumT* b,
        NumT* db
    ) {
        NumT b0 = b[0];
        NumT b1 = b[1];
        NumT b2 = b[2];
        NumT b3 = b[3];

        da[1] += dout[0] * b1 + dout[1] * b0 - dout[2] * b3 + dout[3] * b2;
        da[2] += dout[0] * b2 + dout[1] * b3 + dout[2] * b0 - dout[3] * b1;
        da[3] += dout[0] * b3 - dout[1] * b2 + dout[2] * b1 + dout[3] * b0;

        db[0] = dout[1] * a[1] + dout[2] * a[2] + dout[3] * a[3];
        db[1] = dout[0] * a[1] + dout[2] * a[3] - dout[3] * a[2];
        db[2] = dout[0] * a[2] - dout[1] * a[3] + dout[3] * a[1];
        db[3] = dout[0] * a[3] + dout[1] * a[2] - dout[2] * a[1];
    }

    template <class NumT>
    void add(
        [[maybe_unused]] size_t,
        NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        double scalar
    ) {
        auto x = scalar_as_num<NumT>(scalar);
        a[0] += b[0] * x;
        a[1] += b[1] * x;
        a[2] += b[2] * x;
        a[3] += b[3] * x;
    }

    template <class NumT>
    void scale( [[maybe_unused]] size_t, NumT* MAGNUS_RESTRICT a, double scalar ) {
        auto x = scalar_as_num<NumT>(scalar);
        a[0] *= x;
        a[1] *= x;
        a[2] *= x;
        a[3] *= x;
    }

    template <class NumT>
    void wcopy( size_t total, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
        for ( size_t i = 0; i < total; i += 4 ) {
            dst[i] = src[i];
            dst[i + 1] = src[i + 1];
            dst[i + 2] = src[i + 2];
            dst[i + 3] = src[i + 3];
        }
    }

    template <class NumT>
    void wzero( size_t total, NumT* MAGNUS_RESTRICT dst ) {
        for (size_t i = 0; i < total; ++i) dst[i] = NumT{0};     
    }

    template <class NumT>
    void wadd( size_t total, NumT* MAGNUS_RESTRICT dst, const NumT* MAGNUS_RESTRICT src ) {
        for (size_t i = 0; i < total; ++i) dst[i] += src[i];
    }

    template <class NumT>
    void copy( [[maybe_unused]] size_t, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
    }

    template <class NumT>
    void zero( [[maybe_unused]] size_t, NumT* MAGNUS_RESTRICT dst ) {
        dst[0] = NumT{0};
        dst[1] = NumT{0};
        dst[2] = NumT{0};
        dst[3] = NumT{0};
    }

    template <class NumT>
    void sample_update(
        [[maybe_unused]] size_t dim,
        size_t len,
        const NumT* MAGNUS_RESTRICT A,
        NumT* MAGNUS_RESTRICT Y,
        const NumT* MAGNUS_RESTRICT total,
        double shift,
        [[maybe_unused]] NumT* MAGNUS_RESTRICT temp
    ) {
        auto x = scalar_as_num<NumT>(shift);

        for (size_t i = 0; i < len; ++i) {
            const NumT* MAGNUS_RESTRICT a = A + i * 4;
            NumT* MAGNUS_RESTRICT y = Y + i * 4;

            NumT b0 = y[0] + total[0] * x;
            NumT b1 = y[1] + total[1] * x;
            NumT b2 = y[2] + total[2] * x;
            NumT b3 = y[3] + total[3] * x;

            y[0] = a[1] * b1 + a[2] * b2 + a[3] * b3;
            y[1] = b0 * a[1] + a[2] * b3 - a[3] * b2;
            y[2] = b0 * a[2] + a[3] * b1 - a[1] * b3;
            y[3] = b0 * a[3] + a[1] * b2 - a[2] * b1;
        }
    }

    template <class NumT>
    void sample_update_vjp(
        [[maybe_unused]] size_t dim,
        size_t len,
        NumT* MAGNUS_RESTRICT dA,
        const NumT* MAGNUS_RESTRICT A,
        NumT* MAGNUS_RESTRICT barY,
        const NumT* MAGNUS_RESTRICT prefix,
        double shift,
        NumT* MAGNUS_RESTRICT temp
    ) {
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

            da[1] += g0 * b1 + g1 * b0 - g2 * b3 + g3 * b2;
            da[2] += g0 * b2 + g1 * b3 + g2 * b0 - g3 * b1;
            da[3] += g0 * b3 - g1 * b2 + g2 * b1 + g3 * b0;

            temp[0] = g1 * a[1] + g2 * a[2] + g3 * a[3];
            temp[1] = g0 * a[1] + g2 * a[3] - g3 * a[2];
            temp[2] = g0 * a[2] - g1 * a[3] + g3 * a[1];
            temp[3] = g0 * a[3] + g1 * a[2] - g2 * a[1];
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
    }

    template <class NumT>
    using Policy = GenericMatrixPolicy<NumT, SpaceCurve::matmul<NumT>, SpaceCurve::matmul_vjp<NumT>, SpaceCurve::add<NumT>, SpaceCurve::scale<NumT>, SpaceCurve::copy<NumT>, SpaceCurve::wcopy<NumT>, SpaceCurve::zero<NumT>, SpaceCurve::wzero<NumT>, SpaceCurve::wadd<NumT>, SpaceCurve::sample_update<NumT>, SpaceCurve::sample_update_vjp<NumT>>;

}

#endif
