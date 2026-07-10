#ifndef __INTEGRATION_BOOLE_HPP__
#define __INTEGRATION_BOOLE_HPP__
#include "integrate.hpp"

namespace Magnus {

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class BooleIntegrator {
        DynMatrixSpan<NumT, MatPolicyT, AllocatorT> scratch;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

        static constexpr size_t divisibility_requirement() { return 4; }
        static constexpr size_t memory_requirement() { return 5; }

        BooleIntegrator(size_t dim, const AllocatorT& alloc = AllocatorT()) :
            scratch(dim, 5, alloc) {}

        void prefix(matrix_span_t& A, double dt) {
            size_t len = A.length();

            matrix_t f0 = scratch[0];
            matrix_t f1 = scratch[1];
            matrix_t f2 = scratch[2];
            matrix_t f3 = scratch[3];
            matrix_t f4 = scratch[4];

            f0.copy_from(A[0]);
            A[0].zero();

            size_t start = 0;
            while (start + 1 < len) {
                size_t block_len = std::min<size_t>(4, len - start - 1);
                matrix_t base = A[start];

                f1.copy_from(A[start + 1]);
                A[start + 1]
                    .scale(dt / 2)
                    .add(f0, dt / 2)
                    .add(base);

                f2.copy_from(A[start + 2]);
                A[start + 2]
                    .scale(dt / 3)
                    .add(f0, dt / 3)
                    .add(f1, 4 * dt / 3)
                    .add(base);

                f3.copy_from(A[start + 3]);
                A[start + 3]
                    .scale(3 * dt / 8)
                    .add(f0, 3 * dt / 8)
                    .add(f1, 9 * dt / 8)
                    .add(f2, 9 * dt / 8)
                    .add(base);

                f4.copy_from(A[start + 4]);
                A[start + 4]
                    .scale(14 * dt / 45)
                    .add(f0, 14 * dt / 45)
                    .add(f1, 64 * dt / 45)
                    .add(f2, 24 * dt / 45)
                    .add(f3, 64 * dt / 45)
                    .add(base);

                f0.copy_from(f4);
                start += 4;
            }
        }

        void sum(matrix_span_t& A, matrix_t& out, double dt) {
            size_t len = A.length();

            size_t intervals = len - 1;
            size_t boole_intervals = intervals - intervals % 4;

            for (size_t i = 0; i < boole_intervals; i += 4) {
                out.add(A[i], 14 * dt / 45);
                out.add(A[i + 1], 64 * dt / 45);
                out.add(A[i + 2], 24 * dt / 45);
                out.add(A[i + 3], 64 * dt / 45);
                out.add(A[i + 4], 14 * dt / 45);
            }

            size_t tail = intervals - boole_intervals;
            size_t start = boole_intervals;

            if (tail == 1) {
                out.add(A[start], dt / 2);
                out.add(A[start + 1], dt / 2);
            } else if (tail == 2) {
                out.add(A[start], dt / 3);
                out.add(A[start + 1], 4 * dt / 3);
                out.add(A[start + 2], dt / 3);
            } else if (tail == 3) {
                out.add(A[start], 3 * dt / 8);
                out.add(A[start + 1], 9 * dt / 8);
                out.add(A[start + 2], 9 * dt / 8);
                out.add(A[start + 3], 3 * dt / 8);
            }
        }

        matrix_t borrow_scratch() {
            return scratch[0];
        }
    };


}

#endif