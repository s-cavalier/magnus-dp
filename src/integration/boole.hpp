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

        MAGNUS_ALWAYS_INLINE void prefix(matrix_span_t& A, double dt) {
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
                A[start + 1].linear_combination(
                    dt / 2,
                    f0.term(dt / 2),
                    base.term(1.0)
                );

                f2.copy_from(A[start + 2]);
                A[start + 2].linear_combination(
                    dt / 3,
                    f0.term(dt / 3),
                    f1.term(4 * dt / 3),
                    base.term(1.0)
                );

                f3.copy_from(A[start + 3]);
                A[start + 3].linear_combination(
                    3 * dt / 8,
                    f0.term(3 * dt / 8),
                    f1.term(9 * dt / 8),
                    f2.term(9 * dt / 8),
                    base.term(1.0)
                );

                f4.copy_from(A[start + 4]);
                A[start + 4].linear_combination(
                    14 * dt / 45,
                    f0.term(14 * dt / 45),
                    f1.term(64 * dt / 45),
                    f2.term(24 * dt / 45),
                    f3.term(64 * dt / 45),
                    base.term(1.0)
                );

                f0.copy_from(f4);
                start += 4;
            }
        }

        void sum(matrix_span_t& A, matrix_t& out, double dt) {
            size_t len = A.length();

            size_t intervals = len - 1;
            size_t boole_intervals = intervals - intervals % 4;

            for (size_t i = 0; i < boole_intervals; i += 4) {
                out.linear_combination(
                    1.0,
                    A[i].term(14 * dt / 45),
                    A[i + 1].term(64 * dt / 45),
                    A[i + 2].term(24 * dt / 45),
                    A[i + 3].term(64 * dt / 45),
                    A[i + 4].term(14 * dt / 45)
                );
            }

            size_t tail = intervals - boole_intervals;
            size_t start = boole_intervals;

            if (tail == 1) {
                out.linear_combination(
                    1.0,
                    A[start].term(dt / 2),
                    A[start + 1].term(dt / 2)
                );
            } else if (tail == 2) {
                out.linear_combination(
                    1.0,
                    A[start].term(dt / 3),
                    A[start + 1].term(4 * dt / 3),
                    A[start + 2].term(dt / 3)
                );
            } else if (tail == 3) {
                out.linear_combination(
                    1.0,
                    A[start].term(3 * dt / 8),
                    A[start + 1].term(9 * dt / 8),
                    A[start + 2].term(9 * dt / 8),
                    A[start + 3].term(3 * dt / 8)
                );
            }
        }

        void prefix_vjp(matrix_span_t& A, double dt) {
            matrix_t pending = scratch[0];
            matrix_t g1 = scratch[1];
            matrix_t g2 = scratch[2];
            matrix_t g3 = scratch[3];
            matrix_t g4 = scratch[4];
            pending.zero();

            size_t start = A.length() - 5;
            while (true) {
                g1.copy_from(A[start + 1]);
                g2.copy_from(A[start + 2]);
                g3.copy_from(A[start + 3]);
                g4.copy_from(A[start + 4]);

                A[start].linear_combination(
                    1.0,
                    g1.term(1.0),
                    g2.term(1.0),
                    g3.term(1.0),
                    g4.term(1.0)
                );
                A[start + 1].linear_combination(
                    dt / 2,
                    g2.term(4 * dt / 3),
                    g3.term(9 * dt / 8),
                    g4.term(64 * dt / 45)
                );
                A[start + 2].linear_combination(
                    dt / 3,
                    g3.term(9 * dt / 8),
                    g4.term(24 * dt / 45)
                );
                A[start + 3].linear_combination(
                    3 * dt / 8,
                    g4.term(64 * dt / 45)
                );
                A[start + 4].linear_combination(
                    14 * dt / 45,
                    pending.term(1.0)
                );

                pending.linear_combination(
                    0.0,
                    g1.term(dt / 2),
                    g2.term(dt / 3),
                    g3.term(3 * dt / 8),
                    g4.term(14 * dt / 45)
                );

                if (start == 0) break;
                start -= 4;
            }
            A[0].copy_from(pending);
        }

        void sum_vjp(matrix_span_t& A, const matrix_t& out, double dt) {
            size_t intervals = A.length() - 1;
            size_t boole_intervals = intervals - intervals % 4;

            if (boole_intervals == 0) {
                A[0].zero();
            } else {
                A[0].linear_combination(0.0, out.term(14 * dt / 45));
                for (size_t i = 0; i < boole_intervals; i += 4) {
                    A[i + 1].linear_combination(0.0, out.term(64 * dt / 45));
                    A[i + 2].linear_combination(0.0, out.term(24 * dt / 45));
                    A[i + 3].linear_combination(0.0, out.term(64 * dt / 45));
                }
                for (size_t i = 4; i < boole_intervals; i += 4) {
                    A[i].linear_combination(0.0, out.term(28 * dt / 45));
                }
                A[boole_intervals].linear_combination(0.0, out.term(14 * dt / 45));
            }

            size_t tail = intervals - boole_intervals;
            size_t start = boole_intervals;
            if (tail == 1) {
                A[start].add(out, dt / 2);
                A[start + 1].linear_combination(0.0, out.term(dt / 2));
            } else if (tail == 2) {
                A[start].add(out, dt / 3);
                A[start + 1].linear_combination(0.0, out.term(4 * dt / 3));
                A[start + 2].linear_combination(0.0, out.term(dt / 3));
            } else if (tail == 3) {
                A[start].add(out, 3 * dt / 8);
                A[start + 1].linear_combination(0.0, out.term(9 * dt / 8));
                A[start + 2].linear_combination(0.0, out.term(9 * dt / 8));
                A[start + 3].linear_combination(0.0, out.term(3 * dt / 8));
            }
        }

        void sum_vjp_add(matrix_span_t& A, const matrix_t& out, double dt) {
            size_t intervals = A.length() - 1;
            size_t boole_intervals = intervals - intervals % 4;

            for (size_t i = 0; i < boole_intervals; i += 4) {
                A[i].add(out, 14 * dt / 45);
                A[i + 1].add(out, 64 * dt / 45);
                A[i + 2].add(out, 24 * dt / 45);
                A[i + 3].add(out, 64 * dt / 45);
                A[i + 4].add(out, 14 * dt / 45);
            }

            size_t tail = intervals - boole_intervals;
            size_t start = boole_intervals;
            if (tail == 1) {
                A[start].add(out, dt / 2);
                A[start + 1].add(out, dt / 2);
            } else if (tail == 2) {
                A[start].add(out, dt / 3);
                A[start + 1].add(out, 4 * dt / 3);
                A[start + 2].add(out, dt / 3);
            } else if (tail == 3) {
                A[start].add(out, 3 * dt / 8);
                A[start + 1].add(out, 9 * dt / 8);
                A[start + 2].add(out, 9 * dt / 8);
                A[start + 3].add(out, 3 * dt / 8);
            }
        }

        matrix_t borrow_scratch() {
            return scratch[0];
        }
    };


}

#endif
