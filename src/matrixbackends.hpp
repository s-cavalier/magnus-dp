#ifndef __MAT_BACKENDS_HPP__
#define __MAT_BACKENDS_HPP__
#include "dispatch.hpp"
#include <cblas.h>
#include <execution>
#include <algorithm>


namespace Magnus {

    // TODO: Replace this with a type trait to avoid the runtime conversion
    template <class NumT>
    auto scalar_as_num(double scalar) {
        if constexpr (requires { typename NumT::value_type; }) {
            return static_cast<typename NumT::value_type>(scalar);
        } else {
            return static_cast<NumT>(scalar);
        }
    }

    template <
        class NumT,
        MatMulKernelT<NumT> mm_kernel,
        MatAddKernelT<NumT> ma_kernel,
        MatCopyKernelT<NumT> cp_kernel
    >
    void generic_sample_update(
        size_t dim,
        size_t len,
        const NumT* MAGNUS_RESTRICT A,
        NumT* MAGNUS_RESTRICT Y,
        const NumT* MAGNUS_RESTRICT total,
        double shift,
        NumT* MAGNUS_RESTRICT temp
    ) {
        size_t mat_size = dim * dim;

        for (size_t i = 0; i < len; ++i) {
            const NumT* MAGNUS_RESTRICT a_i = A + i * mat_size;
            NumT* MAGNUS_RESTRICT y_i = Y + i * mat_size;

            ma_kernel(dim, y_i, total, shift);
            mm_kernel(dim, a_i, y_i, temp);
            cp_kernel(dim, temp, y_i);
        }
    }

    template <
        class NumT,
        MatMulKernelT<NumT> mm_kernel,
        MatAddKernelT<NumT> ma_kernel,
        MatScaleKernelT<NumT> ms_kernel,
        MatCopyKernelT<NumT> cp_kernel,
        MatCopyKernelT<NumT> wcp_kernel,
        MatZeroKernelT<NumT> z_kernel,
        SampleUpdateKernelT<NumT> su_kernel = generic_sample_update<NumT, mm_kernel, ma_kernel, cp_kernel>
    >
    struct GenericMatrixPolicy {
        using numeric_t = NumT;

        static void matmul(size_t dim, const NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, NumT* MAGNUS_RESTRICT out) {
            mm_kernel(dim, a, b, out);
        }

        static void matadd(size_t dim, NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, double scalar = 1.0) {
            ma_kernel(dim, a, b, scalar);
        }

        static void matscale(size_t dim, NumT* MAGNUS_RESTRICT a, double scalar) {
            ms_kernel(dim, a, scalar);
        }

        static void matcopy( size_t dim, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
            cp_kernel(dim, src, dst);
        }

        static void matwcopy( size_t total, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst ) {
            wcp_kernel(total, src, dst);
        }

        static void matzero( size_t dim, NumT* MAGNUS_RESTRICT dst ) {
            z_kernel(dim, dst);
        }

        static void sample_update(
            size_t dim,
            size_t len,
            const NumT* MAGNUS_RESTRICT A,
            NumT* MAGNUS_RESTRICT Y,
            const NumT* MAGNUS_RESTRICT total,
            double shift,
            NumT* MAGNUS_RESTRICT temp
        ) {
            su_kernel(dim, len, A, Y, total, shift, temp);
        }

    };

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

    template <class NumT>
    void manual_matmul( size_t dim, const NumT* MAGNUS_RESTRICT a, const NumT* MAGNUS_RESTRICT b, NumT* MAGNUS_RESTRICT out ) {
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                out[i * dim + j] = 0;

                for (size_t k = 0; k < dim; ++k) out[i * dim + j] += a[i * dim + k] * b[k * dim + j];
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
    void manual_matcopy(size_t dim, const NumT* MAGNUS_RESTRICT src, NumT* MAGNUS_RESTRICT dst) {
        manual_matwcopy(dim * dim, src, dst);
    }

    template <class NumT>
    void manual_matzero(size_t dim, NumT* MAGNUS_RESTRICT dst) {
        std::fill_n(dst, dim * dim, NumT{0});
    }

    template <class NumT>
    using ManualPolicy = GenericMatrixPolicy<NumT, manual_matmul<NumT>, manual_matadd<NumT>, manual_matscale<NumT>, manual_matcopy<NumT>, manual_matwcopy<NumT>, manual_matzero<NumT>>;

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

namespace SpaceCurve {

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
    using Policy = GenericMatrixPolicy<NumT, SpaceCurve::matmul<NumT>, SpaceCurve::add<NumT>, SpaceCurve::scale<NumT>, SpaceCurve::copy<NumT>, SpaceCurve::wcopy<NumT>, SpaceCurve::zero<NumT>, SpaceCurve::sample_update<NumT>>;

}

    template <size_t N>
    struct FixedBackend {
        template <Numeric NumT>
        using type = FixedDimPolicy<NumT, N>;

        static inline constexpr auto dispatch_name_storage = make_integral_name<N>("Fixed", "");

        static constexpr std::string_view dispatch_name() {
            return {
                dispatch_name_storage.data(),
                dispatch_name_storage.size()
            };
        }

        static bool valid(const Params& p) { return p.dim == N; }
        static bool use(const Params& p) { return valid(p); }
    };

    struct ManualBackend {
        template <Numeric NumT>
        using type = ManualPolicy<NumT>;

        static constexpr std::string_view dispatch_name() { return "Manual"; }

        static bool valid([[maybe_unused]] const Params&) { return true; }
        static bool use( const Params& p ) { return p.dim < 64; }
    };

    struct BlasBackend {
        template <Numeric NumT>
        using type = BlasPolicy<NumT>;

        static constexpr std::string_view dispatch_name() { return "Blas"; }

        static bool valid([[maybe_unused]] const Params&) { return true; }
        static bool use([[maybe_unused]] const Params&) { return true; }
    };

namespace SpaceCurve {

    struct Backend {
        template <Numeric NumT>
        using type = SpaceCurve::Policy<NumT>;

        static constexpr std::string_view dispatch_name() { return "SpaceCurve"; }

        static bool valid(const Params& p) { return p.dim == 2; }
        static bool use([[maybe_unused]] const Params&) { return true; }
    };

}

    using MatrixBackends = type_list<
        FixedBackend<2>,
        ManualBackend,
        BlasBackend,
        SpaceCurve::Backend
    >;

}

#endif
