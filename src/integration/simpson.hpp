#ifndef __INTEGRATION_SIMPSON_HPP__
#define __INTEGRATION_SIMPSON_HPP__
#include "integrate.hpp"

namespace Magnus {

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class SimpsonIntegrator {
        DynMatrixSpan<NumT, MatPolicyT, AllocatorT> scratch;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

        static constexpr size_t divisibility_requirement() { return 2; }
        static constexpr size_t memory_requirement() { return 3; }

        SimpsonIntegrator(size_t dim, const AllocatorT& alloc = AllocatorT()) :
            scratch(dim, 3, alloc) {}

        void prefix(matrix_span_t& A, double dt) {
            size_t len = A.length();
            matrix_t tmp = scratch[0];
            matrix_t prev2 = scratch[1];
            matrix_t prev1 = scratch[2];

            double half_dt = dt / 2;
            double third_dt = dt / 3;

            prev2.copy_from(A[0]);
            A[0].zero();

            prev1.copy_from(A[1]);
            A[1].scale(half_dt).add(prev2, half_dt);

            for (size_t i = 2; i < len; ++i) {
                tmp.copy_from(A[i]);

                if (i % 2 == 0) {
                    A[i]
                        .scale(third_dt)
                        .add(prev2, third_dt)
                        .add(prev1, 4 * third_dt)
                        .add(A[i - 2]);
                } else {
                    A[i]
                        .scale(half_dt)
                        .add(prev1, half_dt)
                        .add(A[i - 1]);
                }

                prev2.copy_from(prev1);
                prev1.copy_from(tmp);
            }
        }

        void sum(matrix_span_t& A, matrix_t& out, double dt) {
            size_t len = A.length();

            size_t simpson_last = (len % 2 == 1) ? len - 1 : len - 2;

            out.add(A[0], dt / 3);
            for (size_t i = 1; i < simpson_last; ++i) {
                out.add(A[i], (i % 2 == 1) ? 4 * dt / 3 : 2 * dt / 3);
            }
            out.add(A[simpson_last], dt / 3);

            if (simpson_last + 1 < len) {
                out.add(A[simpson_last], dt / 2);
                out.add(A[simpson_last + 1], dt / 2);
            }
        }

        matrix_t borrow_scratch() {
            return scratch[0];
        }
    };

}

#endif