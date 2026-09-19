#ifndef __LINALG_FIXED_HPP__
#define __LINALG_FIXED_HPP__
#include "generic.hpp"
#include <array>
#include <poet/poet.hpp>

namespace Magnus {

    template <class NumT, size_t Dim>
    void fixed_dim_matmul(
        [[maybe_unused]] size_t dim,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        NumT* MAGNUS_RESTRICT out
    ) {
        poet::static_for<Dim>([&](auto I)
        {
            poet::static_for<Dim>([&](auto J)
            {
                out[I * Dim + J] = a[I * Dim] * b[J];

                poet::static_for<1, Dim>([&](auto K)
                {
                    out[I * Dim + J] += a[I * Dim + K] * b[K * Dim + J];
                });
            });
        });

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
        poet::static_for<Dim>([&](auto I) {
            poet::static_for<Dim>([&](auto K) {
                NumT value = dout[I * Dim] * b[K * Dim];
                poet::static_for<1, Dim>([&](auto J) {
                    value += dout[I * Dim + J] * b[K * Dim + J];
                });
                da[I * Dim + K] += value;
            });
        });

        poet::static_for<Dim>([&](auto K) {
            poet::static_for<Dim>([&](auto J) {
                NumT value = a[K] * dout[J];
                poet::static_for<1, Dim>([&](auto I) {
                    value += a[I * Dim + K] * dout[I * Dim + J];
                });
                db[K * Dim + J] = value;
            });
        });
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matadd(
        [[maybe_unused]] size_t dim,
        NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        double scalar
    ) {
        auto x = scalar_as_num<NumT>(scalar);
        poet::static_for<Dim * Dim>([&](auto I) { a[I] += b[I] * x; });
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matscale( [[maybe_unused]] size_t dim, NumT* MAGNUS_RESTRICT a, double scalar ) {
        auto x = scalar_as_num<NumT>(scalar);
        poet::static_for<Dim * Dim>([&](auto I) { a[I] *= x; });
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
        poet::static_for<Dim * Dim>([&](auto I) { dst[I] = src[I]; });
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matzero( [[maybe_unused]] size_t, NumT* MAGNUS_RESTRICT dst ) {
        poet::static_for<Dim * Dim>([&](auto I) { dst[I] = NumT{0}; });
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
        static_cast<void>(temp);
        auto x = scalar_as_num<NumT>(shift);

        for (size_t sample = 0; sample < len; ++sample) {
            const NumT* MAGNUS_RESTRICT a = A + sample * Dim * Dim;
            NumT* MAGNUS_RESTRICT y = Y + sample * Dim * Dim;

            std::array<NumT, Dim * Dim> b;
            poet::static_for<Dim * Dim>([&](auto I) {
                b[I] = y[I] + total[I] * x;
            });

            poet::static_for<Dim>([&](auto I) {
                poet::static_for<Dim>([&](auto J) {
                    NumT value = a[I * Dim] * b[J];
                    poet::static_for<1, Dim>([&](auto K) {
                        value += a[I * Dim + K] * b[K * Dim + J];
                    });
                    y[I * Dim + J] = value;
                });
            });
        }
    }

    template <class NumT, size_t Dim>
    void fixed_dim_sample_update_vjp(
        size_t,
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
        const NumT* MAGNUS_RESTRICT total = prefix + last * Dim * Dim;

        auto reverse_sample = [&](size_t sample) {
            NumT* MAGNUS_RESTRICT da = dA + sample * Dim * Dim;
            const NumT* MAGNUS_RESTRICT a = A + sample * Dim * Dim;
            const NumT* MAGNUS_RESTRICT g = barY + sample * Dim * Dim;
            const NumT* MAGNUS_RESTRICT p = prefix + sample * Dim * Dim;

            std::array<NumT, Dim * Dim> b;
            std::array<NumT, Dim * Dim> g_copy;
            poet::static_for<Dim * Dim>([&](auto I) {
                b[I] = p[I] + total[I] * x;
                g_copy[I] = g[I];
            });

            poet::static_for<Dim>([&](auto I) {
                poet::static_for<Dim>([&](auto K) {
                    NumT value = g_copy[I * Dim] * b[K * Dim];
                    poet::static_for<1, Dim>([&](auto J) {
                        value += g_copy[I * Dim + J] * b[K * Dim + J];
                    });
                    da[I * Dim + K] += value;
                });
            });

            poet::static_for<Dim>([&](auto K) {
                poet::static_for<Dim>([&](auto J) {
                    NumT value = a[K] * g_copy[J];
                    poet::static_for<1, Dim>([&](auto I) {
                        value += a[I * Dim + K] * g_copy[I * Dim + J];
                    });
                    temp[K * Dim + J] = value;
                });
            });
        };

        reverse_sample(last);
        NumT* MAGNUS_RESTRICT bar_total = barY + last * Dim * Dim;
        poet::static_for<Dim * Dim>([&](auto I) {
            bar_total[I] = temp[I] * one_plus_x;
        });

        for (size_t sample = last; sample-- > 0;) {
            reverse_sample(sample);
            NumT* MAGNUS_RESTRICT g = barY + sample * Dim * Dim;
            poet::static_for<Dim * Dim>([&](auto I) {
                bar_total[I] += temp[I] * x;
                g[I] = temp[I];
            });
        }
    }

    template <class NumT, size_t Dim>
    using FixedDimPolicy = GenericMatrixPolicy<
        NumT,
        fixed_dim_matmul<NumT, Dim>,
        fixed_dim_matmul_vjp<NumT, Dim>,
        fixed_dim_matadd<NumT, Dim>,
        fixed_dim_matscale<NumT, Dim>,
        fixed_dim_matcopy<NumT, Dim>,
        fixed_dim_matwcopy<NumT, Dim>,
        fixed_dim_matzero<NumT, Dim>,
        fixed_dim_matwzero<NumT, Dim>,
        fixed_dim_matwadd<NumT, Dim>,
        fixed_dim_sample_update<NumT, Dim>,
        fixed_dim_sample_update_vjp<NumT, Dim>
    >;


}

#endif
