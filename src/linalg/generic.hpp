#ifndef __LINALG_GENERIC_HPP__
#define __LINALG_GENERIC_HPP__
#include "matrix.hpp"
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
}

#endif