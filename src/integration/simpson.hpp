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
            A[1].linear_combination(half_dt, prev2.term(half_dt));

            for (size_t i = 2; i < len; ++i) {
                tmp.copy_from(A[i]);

                if (i % 2 == 0) {
                    A[i].linear_combination(
                        third_dt,
                        prev2.term(third_dt),
                        prev1.term(4 * third_dt),
                        A[i - 2].term(1.0)
                    );
                } else {
                    A[i].linear_combination(
                        half_dt,
                        prev1.term(half_dt),
                        A[i - 1].term(1.0)
                    );
                }

                prev2.copy_from(prev1);
                prev1.copy_from(tmp);
            }
        }

        void sum(matrix_span_t& A, matrix_t& out, double dt) {
            size_t len = A.length();

            size_t simpson_last = (len % 2 == 1) ? len - 1 : len - 2;

            out.add(A[0], dt / 3);
            for (size_t i = 1; i + 1 < simpson_last; i += 2) {
                out.linear_combination(
                    1.0,
                    A[i].term(4 * dt / 3),
                    A[i + 1].term(2 * dt / 3)
                );
            }

            if (simpson_last + 1 < len) {
                out.linear_combination(
                    1.0,
                    A[simpson_last - 1].term(4 * dt / 3),
                    A[simpson_last].term(dt / 3 + dt / 2),
                    A[simpson_last + 1].term(dt / 2)
                );
            } else {
                out.linear_combination(
                    1.0,
                    A[simpson_last - 1].term(4 * dt / 3),
                    A[simpson_last].term(dt / 3)
                );
            }
        }

        void prefix_vjp(matrix_span_t& A, double dt) {
            matrix_t bar = scratch[0];
            matrix_t pending = scratch[1];
            matrix_t next_pending = scratch[2];
            double half_dt = dt / 2;
            double third_dt = dt / 3;

            pending.zero();
            next_pending.zero();

            auto reverse_odd = [&](size_t i) {
                bar.copy_from(A[i]);
                A[i - 1].add(bar);
                A[i].linear_combination(half_dt, pending.term(1.0));
                next_pending.add(bar, half_dt);
                pending.zero();
                std::swap(pending, next_pending);
            };

            auto reverse_even = [&](size_t i) {
                bar.copy_from(A[i]);
                A[i - 2].add(bar);
                A[i].linear_combination(third_dt, pending.term(1.0));
                next_pending.add(bar, 4 * third_dt);
                pending.linear_combination(0.0, bar.term(third_dt));
                std::swap(pending, next_pending);
            };

            size_t i = A.length() - 1;
            if (i % 2 == 1) {
                reverse_odd(i);
                --i;
            }

            while (i > 2) {
                reverse_even(i);
                reverse_odd(i - 1);
                i -= 2;
            }
            reverse_even(2);

            bar.copy_from(A[1]);
            A[1].linear_combination(half_dt, pending.term(1.0));
            A[0].linear_combination(
                0.0,
                bar.term(half_dt),
                next_pending.term(1.0)
            );
        }

        void sum_vjp(matrix_span_t& A, const matrix_t& out, double dt) {
            size_t len = A.length();
            size_t simpson_last = (len % 2 == 1) ? len - 1 : len - 2;

            A[0].linear_combination(0.0, out.term(dt / 3));
            for (size_t i = 1; i < simpson_last; i += 2) {
                A[i].linear_combination(0.0, out.term(4 * dt / 3));
            }
            for (size_t i = 2; i < simpson_last; i += 2) {
                A[i].linear_combination(0.0, out.term(2 * dt / 3));
            }
            A[simpson_last].linear_combination(0.0, out.term(dt / 3));

            if (simpson_last + 1 < len) {
                A[simpson_last].add(out, dt / 2);
                A[simpson_last + 1].linear_combination(0.0, out.term(dt / 2));
            }
        }

        void sum_vjp_add(matrix_span_t& A, const matrix_t& out, double dt) {
            size_t len = A.length();
            size_t simpson_last = (len % 2 == 1) ? len - 1 : len - 2;

            A[0].add(out, dt / 3);
            for (size_t i = 1; i < simpson_last; i += 2) A[i].add(out, 4 * dt / 3);
            for (size_t i = 2; i < simpson_last; i += 2) A[i].add(out, 2 * dt / 3);
            A[simpson_last].add(out, dt / 3);

            if (simpson_last + 1 < len) {
                A[simpson_last].add(out, dt / 2);
                A[simpson_last + 1].add(out, dt / 2);
            }
        }

        matrix_t borrow_scratch() {
            return scratch[0];
        }
    };

}

#endif
