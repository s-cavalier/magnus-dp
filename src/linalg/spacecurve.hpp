#ifndef __LINALG_SPACECURVE_HPP__
#define __LINALG_SPACECURVE_HPP__
#include "generic.hpp"
#include "poet/core/cpu_info.hpp"
#include <array>
#include <poet/poet.hpp>

#if MAGNUS_IS_X86
#include <immintrin.h>
#endif

namespace Magnus::SpaceCurve {

    inline constexpr size_t vector_dim = 3;
    inline constexpr size_t storage_size = vector_dim + 1;

    // Until an AVX-512 implementation exists, use the AVX2 kernel for an
    // AVX-512 target as well. AVX-512 targets support AVX2 instructions, and
    // selecting detected_isa() directly would otherwise fall through to the
    // scalar implementation.
    consteval poet::instruction_set default_isa() {
        if constexpr (poet::detected_isa() == poet::instruction_set::avx_512) {
            return poet::instruction_set::avx2;
        }
        return poet::detected_isa();
    }

    template <class NumT>
    MAGNUS_ALWAYS_INLINE void wedge_product(
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        NumT* MAGNUS_RESTRICT out
    ) {
        out[0] = a[1] * b[1];
        poet::static_for<1, vector_dim>([&](auto Axis) {
            constexpr size_t component = Axis + 1;
            out[0] += a[component] * b[component];
        });

        poet::static_for<vector_dim>([&](auto Axis) {
            constexpr size_t component = Axis + 1;
            constexpr size_t next = (Axis + 1) % vector_dim + 1;
            constexpr size_t previous = (Axis + 2) % vector_dim + 1;

            out[component] = -b[0] * a[component];
            out[component] += a[next] * b[previous];
            out[component] -= a[previous] * b[next];
        });
    }

    template <class NumT>
    MAGNUS_ALWAYS_INLINE void wedge_product_vjp(
        NumT* MAGNUS_RESTRICT da,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT dout,
        const NumT* MAGNUS_RESTRICT b,
        NumT* MAGNUS_RESTRICT db
    ) {
        poet::static_for<vector_dim>([&](auto Axis) {
            constexpr size_t component = Axis + 1;
            constexpr size_t next = (Axis + 1) % vector_dim + 1;
            constexpr size_t previous = (Axis + 2) % vector_dim + 1;

            NumT value = dout[0] * b[component];
            value -= dout[component] * b[0];
            value -= dout[next] * b[previous];
            value += dout[previous] * b[next];
            da[component] += value;
        });

        db[0] = -dout[1] * a[1];
        poet::static_for<1, vector_dim>([&](auto Axis) {
            constexpr size_t component = Axis + 1;
            db[0] -= dout[component] * a[component];
        });

        poet::static_for<vector_dim>([&](auto Axis) {
            constexpr size_t component = Axis + 1;
            constexpr size_t next = (Axis + 1) % vector_dim + 1;
            constexpr size_t previous = (Axis + 2) % vector_dim + 1;

            db[component] = dout[0] * a[component];
            db[component] += dout[next] * a[previous];
            db[component] -= dout[previous] * a[next];
        });
    }

    template <class NumT>
    void matmul(
        [[maybe_unused]] size_t,
        const NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        NumT* MAGNUS_RESTRICT c
    ) {
        wedge_product(a, b, c);
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
        wedge_product_vjp(da, a, dout, b, db);
    }

    template <class NumT>
    void add(
        [[maybe_unused]] size_t,
        NumT* MAGNUS_RESTRICT a,
        const NumT* MAGNUS_RESTRICT b,
        double scalar
    ) {
        auto x = scalar_as_num<NumT>(scalar);
        poet::static_for<storage_size>([&](auto I) { a[I] += b[I] * x; });
    }

    template <class NumT>
    void scale( [[maybe_unused]] size_t, NumT* MAGNUS_RESTRICT a, double scalar ) {
        auto x = scalar_as_num<NumT>(scalar);
        poet::static_for<storage_size>([&](auto I) { a[I] *= x; });
    }

    template <class NumT>
    void wcopy( size_t total, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
        for ( size_t i = 0; i < total; i += storage_size ) {
            poet::static_for<storage_size>([&](auto I) { dst[i + I] = src[i + I]; });
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
        poet::static_for<storage_size>([&](auto I) { dst[I] = src[I]; });
    }

    template <class NumT>
    void zero( [[maybe_unused]] size_t, NumT* MAGNUS_RESTRICT dst ) {
        poet::static_for<storage_size>([&](auto I) { dst[I] = NumT{0}; });
    }

    template <class NumT, poet::instruction_set Arch = default_isa()>
    void sample_update(
        [[maybe_unused]] size_t dim,
        size_t len,
        const NumT* MAGNUS_RESTRICT A,
        NumT* MAGNUS_RESTRICT Y,
        const NumT* MAGNUS_RESTRICT total,
        double shift,
        [[maybe_unused]] NumT* MAGNUS_RESTRICT temp
    ) {
#if MAGNUS_IS_X86
        if constexpr ( Arch == poet::instruction_set::avx2 && std::same_as<NumT, double>) {
            const __m256d shift_v = _mm256_set1_pd(shift);
            size_t sample = 0;

            auto transpose4 = []( __m256d& row0, __m256d& row1, __m256d& row2, __m256d& row3) {
                const __m256d pair01lo = _mm256_unpacklo_pd(row0, row1);
                const __m256d pair01hi = _mm256_unpackhi_pd(row0, row1);
                const __m256d pair23lo = _mm256_unpacklo_pd(row2, row3);
                const __m256d pair23hi = _mm256_unpackhi_pd(row2, row3);

                row0 = _mm256_permute2f128_pd(pair01lo, pair23lo, 0x20);
                row1 = _mm256_permute2f128_pd(pair01hi, pair23hi, 0x20);
                row2 = _mm256_permute2f128_pd(pair01lo, pair23lo, 0x31);
                row3 = _mm256_permute2f128_pd(pair01hi, pair23hi, 0x31);
            };

            for (; sample + 4 <= len; sample += 4) {
                __m256d a0 = _mm256_loadu_pd(A + storage_size * sample);
                __m256d a1 = _mm256_loadu_pd(A + storage_size * sample + 4);
                __m256d a2 = _mm256_loadu_pd(A + storage_size * sample + 8);
                __m256d a3 = _mm256_loadu_pd(A + storage_size * sample + 12);
                transpose4(a0, a1, a2, a3);

                __m256d b0 = _mm256_loadu_pd(Y + storage_size * sample);
                __m256d b1 = _mm256_loadu_pd(Y + storage_size * sample + 4);
                __m256d b2 = _mm256_loadu_pd(Y + storage_size * sample + 8);
                __m256d b3 = _mm256_loadu_pd(Y + storage_size * sample + 12);
                transpose4(b0, b1, b2, b3);

                b0 = _mm256_fmadd_pd(_mm256_broadcast_sd(total), shift_v, b0);
                b1 = _mm256_fmadd_pd(_mm256_broadcast_sd(total + 1), shift_v, b1);
                b2 = _mm256_fmadd_pd(_mm256_broadcast_sd(total + 2), shift_v, b2);
                b3 = _mm256_fmadd_pd(_mm256_broadcast_sd(total + 3), shift_v, b3);

                __m256d out0 = _mm256_fmadd_pd(a3, b3, _mm256_fmadd_pd(a2, b2, _mm256_mul_pd(a1, b1)));
                __m256d out1 = _mm256_sub_pd(_mm256_fmsub_pd(a2, b3, _mm256_mul_pd(a3, b2)), _mm256_mul_pd(b0, a1));
                __m256d out2 = _mm256_sub_pd(_mm256_fmsub_pd(a3, b1, _mm256_mul_pd(a1, b3)), _mm256_mul_pd(b0, a2));
                __m256d out3 = _mm256_sub_pd(_mm256_fmsub_pd(a1, b2, _mm256_mul_pd(a2, b1)), _mm256_mul_pd(b0, a3));
                transpose4(out0, out1, out2, out3);

                _mm256_storeu_pd(Y + storage_size * sample, out0);
                _mm256_storeu_pd(Y + storage_size * sample + 4, out1);
                _mm256_storeu_pd(Y + storage_size * sample + 8, out2);
                _mm256_storeu_pd(Y + storage_size * sample + 12, out3);
            }

            for (; sample < len; ++sample) {
                const double* MAGNUS_RESTRICT a = A + storage_size * sample;
                double* MAGNUS_RESTRICT y = Y + storage_size * sample;
                const double b0 = y[0] + total[0] * shift;
                const double b1 = y[1] + total[1] * shift;
                const double b2 = y[2] + total[2] * shift;
                const double b3 = y[3] + total[3] * shift;

                y[0] = a[1] * b1 + a[2] * b2 + a[3] * b3;
                y[1] = -b0 * a[1] + a[2] * b3 - a[3] * b2;
                y[2] = -b0 * a[2] + a[3] * b1 - a[1] * b3;
                y[3] = -b0 * a[3] + a[1] * b2 - a[2] * b1;
            }
            return;
        }
#endif

        const NumT x = scalar_as_num<NumT>(shift);
        for (size_t sample = 0; sample < len; ++sample) {
            const NumT* MAGNUS_RESTRICT a = A + sample * storage_size;
            NumT* MAGNUS_RESTRICT y = Y + sample * storage_size;
            std::array<NumT, storage_size> b;
            poet::static_for<storage_size>([&](auto I) {
                b[I] = y[I] + total[I] * x;
            });
            wedge_product(a, b.data(), y);
        }
    }

    template <class NumT, poet::instruction_set Arch = default_isa()>
    MAGNUS_ALWAYS_INLINE void sample_update_vjp(
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
        const NumT* MAGNUS_RESTRICT total = prefix + last * storage_size;

        auto reverse_sample = [&](size_t sample) {
            NumT* MAGNUS_RESTRICT da = dA + sample * storage_size;
            const NumT* MAGNUS_RESTRICT a = A + sample * storage_size;
            const NumT* MAGNUS_RESTRICT g = barY + sample * storage_size;
            const NumT* MAGNUS_RESTRICT p = prefix + sample * storage_size;

            std::array<NumT, storage_size> b;
            poet::static_for<storage_size>([&](auto I) {
                b[I] = p[I] + total[I] * x;
            });

            poet::static_for<vector_dim>([&](auto Axis) {
                constexpr size_t component = Axis + 1;
                constexpr size_t next = (Axis + 1) % vector_dim + 1;
                constexpr size_t previous = (Axis + 2) % vector_dim + 1;

                da[component] +=
                    g[0] * b[component]
                    - g[component] * b[0]
                    - g[next] * b[previous]
                    + g[previous] * b[next];
            });

            temp[0] = -g[1] * a[1];
            poet::static_for<1, vector_dim>([&](auto Axis) {
                constexpr size_t component = Axis + 1;
                temp[0] -= g[component] * a[component];
            });

            poet::static_for<vector_dim>([&](auto Axis) {
                constexpr size_t component = Axis + 1;
                constexpr size_t next = (Axis + 1) % vector_dim + 1;
                constexpr size_t previous = (Axis + 2) % vector_dim + 1;

                temp[component] =
                    g[0] * a[component]
                    + g[next] * a[previous]
                    - g[previous] * a[next];
            });
        };

#if MAGNUS_IS_X86
        if constexpr (Arch == poet::instruction_set::avx2 && std::same_as<NumT, double>) {
            const __m256d shift_v = _mm256_set1_pd(shift);

            auto transpose4 = [](__m256d& row0, __m256d& row1, __m256d& row2, __m256d& row3) {
                const __m256d pair01lo = _mm256_unpacklo_pd(row0, row1);
                const __m256d pair01hi = _mm256_unpackhi_pd(row0, row1);
                const __m256d pair23lo = _mm256_unpacklo_pd(row2, row3);
                const __m256d pair23hi = _mm256_unpackhi_pd(row2, row3);

                row0 = _mm256_permute2f128_pd(pair01lo, pair23lo, 0x20);
                row1 = _mm256_permute2f128_pd(pair01hi, pair23hi, 0x20);
                row2 = _mm256_permute2f128_pd(pair01lo, pair23lo, 0x31);
                row3 = _mm256_permute2f128_pd(pair01hi, pair23hi, 0x31);
            };

            reverse_sample(last);
            NumT* MAGNUS_RESTRICT bar_total = barY + last * storage_size;
            poet::static_for<storage_size>([&](auto I) {
                bar_total[I] = temp[I] * one_plus_x;
            });

            size_t sample = last;
            __m256d bar_total0 = _mm256_setzero_pd();
            __m256d bar_total1 = _mm256_setzero_pd();
            __m256d bar_total2 = _mm256_setzero_pd();
            __m256d bar_total3 = _mm256_setzero_pd();
            for (; sample >= 4; sample -= 4) {
                const size_t base = sample - 4;

                __m256d a0 = _mm256_loadu_pd(A + storage_size * base);
                __m256d a1 = _mm256_loadu_pd(A + storage_size * base + 4);
                __m256d a2 = _mm256_loadu_pd(A + storage_size * base + 8);
                __m256d a3 = _mm256_loadu_pd(A + storage_size * base + 12);
                transpose4(a0, a1, a2, a3);

                __m256d g0 = _mm256_loadu_pd(barY + storage_size * base);
                __m256d g1 = _mm256_loadu_pd(barY + storage_size * base + 4);
                __m256d g2 = _mm256_loadu_pd(barY + storage_size * base + 8);
                __m256d g3 = _mm256_loadu_pd(barY + storage_size * base + 12);
                transpose4(g0, g1, g2, g3);

                __m256d b0 = _mm256_loadu_pd(prefix + storage_size * base);
                __m256d b1 = _mm256_loadu_pd(prefix + storage_size * base + 4);
                __m256d b2 = _mm256_loadu_pd(prefix + storage_size * base + 8);
                __m256d b3 = _mm256_loadu_pd(prefix + storage_size * base + 12);
                transpose4(b0, b1, b2, b3);
                b0 = _mm256_fmadd_pd(_mm256_broadcast_sd(total), shift_v, b0);
                b1 = _mm256_fmadd_pd(_mm256_broadcast_sd(total + 1), shift_v, b1);
                b2 = _mm256_fmadd_pd(_mm256_broadcast_sd(total + 2), shift_v, b2);
                b3 = _mm256_fmadd_pd(_mm256_broadcast_sd(total + 3), shift_v, b3);

                __m256d da0 = _mm256_setzero_pd();
                __m256d da1 = _mm256_mul_pd(g0, b1);
                da1 = _mm256_fnmadd_pd(g1, b0, da1);
                da1 = _mm256_fnmadd_pd(g2, b3, da1);
                da1 = _mm256_fmadd_pd(g3, b2, da1);

                __m256d da2 = _mm256_mul_pd(g0, b2);
                da2 = _mm256_fnmadd_pd(g2, b0, da2);
                da2 = _mm256_fnmadd_pd(g3, b1, da2);
                da2 = _mm256_fmadd_pd(g1, b3, da2);

                __m256d da3 = _mm256_mul_pd(g0, b3);
                da3 = _mm256_fnmadd_pd(g3, b0, da3);
                da3 = _mm256_fnmadd_pd(g1, b2, da3);
                da3 = _mm256_fmadd_pd(g2, b1, da3);

                __m256d temp0 = _mm256_mul_pd(g1, a1);
                temp0 = _mm256_fmadd_pd(g2, a2, temp0);
                temp0 = _mm256_fnmadd_pd(g3, a3, _mm256_sub_pd(_mm256_setzero_pd(), temp0));

                __m256d temp1 = _mm256_sub_pd(
                    _mm256_fmadd_pd(g2, a3, _mm256_mul_pd(g0, a1)),
                    _mm256_mul_pd(g3, a2)
                );
                __m256d temp2 = _mm256_sub_pd(
                    _mm256_fmadd_pd(g3, a1, _mm256_mul_pd(g0, a2)),
                    _mm256_mul_pd(g1, a3)
                );
                __m256d temp3 = _mm256_sub_pd(
                    _mm256_fmadd_pd(g1, a2, _mm256_mul_pd(g0, a3)),
                    _mm256_mul_pd(g2, a1)
                );

                transpose4(da0, da1, da2, da3);
                _mm256_storeu_pd(dA + storage_size * base, _mm256_add_pd(_mm256_loadu_pd(dA + storage_size * base), da0));
                _mm256_storeu_pd(dA + storage_size * base + 4, _mm256_add_pd(_mm256_loadu_pd(dA + storage_size * base + 4), da1));
                _mm256_storeu_pd(dA + storage_size * base + 8, _mm256_add_pd(_mm256_loadu_pd(dA + storage_size * base + 8), da2));
                _mm256_storeu_pd(dA + storage_size * base + 12, _mm256_add_pd(_mm256_loadu_pd(dA + storage_size * base + 12), da3));

                bar_total0 = _mm256_fmadd_pd(temp0, shift_v, bar_total0);
                bar_total1 = _mm256_fmadd_pd(temp1, shift_v, bar_total1);
                bar_total2 = _mm256_fmadd_pd(temp2, shift_v, bar_total2);
                bar_total3 = _mm256_fmadd_pd(temp3, shift_v, bar_total3);

                transpose4(temp0, temp1, temp2, temp3);
                _mm256_storeu_pd(barY + storage_size * base, temp0);
                _mm256_storeu_pd(barY + storage_size * base + 4, temp1);
                _mm256_storeu_pd(barY + storage_size * base + 8, temp2);
                _mm256_storeu_pd(barY + storage_size * base + 12, temp3);
            }

            alignas(32) double bar_total0_lanes[4];
            alignas(32) double bar_total1_lanes[4];
            alignas(32) double bar_total2_lanes[4];
            alignas(32) double bar_total3_lanes[4];
            _mm256_store_pd(bar_total0_lanes, bar_total0);
            _mm256_store_pd(bar_total1_lanes, bar_total1);
            _mm256_store_pd(bar_total2_lanes, bar_total2);
            _mm256_store_pd(bar_total3_lanes, bar_total3);
            poet::static_for<4>([&](auto I) {
                bar_total[0] += bar_total0_lanes[I];
                bar_total[1] += bar_total1_lanes[I];
                bar_total[2] += bar_total2_lanes[I];
                bar_total[3] += bar_total3_lanes[I];
            });

            for (; sample > 0; --sample) {
                const size_t current = sample - 1;
                reverse_sample(current);
                NumT* MAGNUS_RESTRICT g = barY + current * storage_size;
                poet::static_for<storage_size>([&](auto I) {
                    bar_total[I] += temp[I] * x;
                    g[I] = temp[I];
                });
            }
            return;
        }
#endif

        reverse_sample(last);
        NumT* MAGNUS_RESTRICT bar_total = barY + last * storage_size;
        poet::static_for<storage_size>([&](auto I) {
            bar_total[I] = temp[I] * one_plus_x;
        });

        for (size_t sample = last; sample-- > 0;) {
            reverse_sample(sample);
            NumT* MAGNUS_RESTRICT g = barY + sample * storage_size;
            poet::static_for<storage_size>([&](auto I) {
                bar_total[I] += temp[I] * x;
            });
            poet::static_for<storage_size>([&](auto I) {
                g[I] = temp[I];
            });
        }
    }

    template <class NumT>
    struct LinearCombine {
        template <class First, class... Pairs>
        MAGNUS_ALWAYS_INLINE static void apply(
            [[maybe_unused]] size_t dim,
            NumT* MAGNUS_RESTRICT dst,
            First&& first,
            Pairs&&... pairs
        ) {
            if constexpr (std::same_as<std::remove_cvref_t<First>, double>) {
                static_assert(sizeof...(Pairs) > 0);
                const NumT initial = scalar_as_num<NumT>(first);
                poet::static_for<storage_size>([&] [[gnu::always_inline]] (auto I) {
                    NumT value = dst[I] * initial;
                    ((value += pairs.first[I] * scalar_as_num<NumT>(pairs.second)), ...);
                    dst[I] = value;
                });
            } else {
                poet::static_for<storage_size>([&] [[gnu::always_inline]] (auto I) {
                    NumT value = first.first[I] * scalar_as_num<NumT>(first.second);
                    ((value += pairs.first[I] * scalar_as_num<NumT>(pairs.second)), ...);
                    dst[I] = value;
                });
            }
        }
    };

    template <class NumT, poet::instruction_set Arch = default_isa()>
    using Policy = GenericMatrixPolicy<
        NumT,
        SpaceCurve::matmul<NumT>,
        SpaceCurve::matmul_vjp<NumT>,
        SpaceCurve::add<NumT>,
        SpaceCurve::scale<NumT>,
        SpaceCurve::copy<NumT>,
        SpaceCurve::wcopy<NumT>,
        SpaceCurve::zero<NumT>,
        SpaceCurve::wzero<NumT>,
        SpaceCurve::wadd<NumT>,
        SpaceCurve::sample_update<NumT, Arch>,
        SpaceCurve::sample_update_vjp<NumT, Arch>,
        LinearCombine<NumT>
    >;

}

#endif
