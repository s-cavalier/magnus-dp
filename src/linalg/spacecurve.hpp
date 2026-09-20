#ifndef __LINALG_SPACECURVE_HPP__
#define __LINALG_SPACECURVE_HPP__
#include "generic.hpp"
#include <array>
#include <poet/poet.hpp>

namespace Magnus::SpaceCurve {

    inline constexpr size_t vector_dim = 3;
    inline constexpr size_t storage_size = vector_dim + 1;

    template <class NumT>
    struct LinearCombination {
        template <std::same_as<std::pair<double, const NumT* MAGNUS_RESTRICT>>... Terms>
        MAGNUS_ALWAYS_INLINE static void invoke(
            [[maybe_unused]] size_t dim,
            NumT* MAGNUS_RESTRICT destination,
            double destination_coefficient,
            Terms&&... terms
        ) {
            const auto destination_scalar = scalar_as_num<NumT>(destination_coefficient);
            poet::static_for<storage_size>([&] [[gnu::always_inline]] (auto I) {
                NumT value = destination[I] * destination_scalar;
                ((value += terms.second[I] * scalar_as_num<NumT>(terms.first)), ...);
                destination[I] = value;
            });
        }
    };

    // Internal values use (scalar, vector) storage. With a representing a
    // tangent vector, the reduced Pauli product is
    //     a wedge b = (dot(a, b), cross(a, b) - b.scalar * a).
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
        SpaceCurve::LinearCombination<NumT>,
        SpaceCurve::sample_update<NumT>,
        SpaceCurve::sample_update_vjp<NumT>
    >;

}

#endif
