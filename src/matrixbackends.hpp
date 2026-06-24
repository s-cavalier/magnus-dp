#ifndef __MAT_BACKENDS_HPP__
#define __MAT_BACKENDS_HPP__
#include "dispatch.hpp"
#include <cblas.h>

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

    template <class NumT, MatMulKernelT<NumT> mm_kernel, MatAddKernelT<NumT> ma_kernel, MatScaleKernelT<NumT> ms_kernel>
    struct GenericMatrixPolicy {
        using numeric_t = NumT;

        static void matmul(size_t dim, const NumT* a, const NumT* b, NumT* out) {
            mm_kernel(dim, a, b, out);
        }

        static void matadd(size_t dim, NumT* a, const NumT* b, double scalar = 1.0) {
            ma_kernel(dim, a, b, scalar);
        }

        static void matscale(size_t dim, NumT* a, double scalar) {
            ms_kernel(dim, a, scalar);
        }
    };

    template <class NumT>
    void blas_matmul(size_t dim, const NumT* a, const NumT* b, NumT* out ) {
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
    void blas_matadd( size_t dim, NumT* a, const NumT* b, double scalar ) {
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
    void blas_matscale(size_t dim, NumT* a, double scalar) {
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
    using BlasPolicy = GenericMatrixPolicy<NumT, blas_matmul<NumT>, blas_matadd<NumT>, blas_matscale<NumT>>;

    template <class NumT>
    void manual_matmul( size_t dim, const NumT* a, const NumT* b, NumT* out ) {
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                out[i * dim + j] = 0;

                for (size_t k = 0; k < dim; ++k) out[i * dim + j] += a[i * dim + k] * b[k * dim + j];
            }
        }

    }

    template <class NumT>
    void manual_matadd(size_t dim, NumT* a, const NumT* b, double scalar) {
        size_t size = dim * dim;
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < size; ++i) a[i] += b[i] * x;
    }

    template <class NumT>
    void manual_matscale(size_t dim, NumT* a, double scalar) {
        size_t size = dim * dim;
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < size; ++i) a[i] *= x;
    }

    template <class NumT>
    using ManualPolicy = GenericMatrixPolicy<NumT, manual_matmul<NumT>, manual_matadd<NumT>, manual_matscale<NumT>>;

    template <class NumT, size_t Dim>
    void fixed_dim_matmul([[maybe_unused]] size_t dim, const NumT* a, const NumT* b, NumT* out) {
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
    void fixed_dim_matadd( [[maybe_unused]] size_t dim, NumT* a, const NumT* b, double scalar ) {
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < Dim * Dim; ++i) a[i] += b[i] * x;
    }

    template <class NumT, size_t Dim>
    void fixed_dim_matscale( [[maybe_unused]] size_t dim, NumT* a, double scalar ) {
        auto x = scalar_as_num<NumT>(scalar);
        for (size_t i = 0; i < Dim * Dim; ++i) a[i] *= x;
    }

    template <class NumT, size_t Dim>
    using FixedDimPolicy = GenericMatrixPolicy<NumT, fixed_dim_matmul<NumT, Dim>, fixed_dim_matadd<NumT, Dim>, fixed_dim_matscale<NumT, Dim>>;

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

    static constexpr auto res = FixedBackend<2>::dispatch_name();

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

    using MatrixBackends = type_list<
        FixedBackend<2>,
        ManualBackend,
        BlasBackend
    >;

}

#endif