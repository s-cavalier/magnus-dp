#ifndef __INTEGRATION_TRAPEZOID_HPP__
#define __INTEGRATION_TRAPEZOID_HPP__
#include "integrate.hpp"

namespace Magnus {

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class TrapezoidalIntegrator {
        DynMatrix<NumT, MatPolicyT, AllocatorT> tmp;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

        static constexpr size_t divisibility_requirement() { return 2; }
        static constexpr size_t memory_requirement() { return 1; }

        TrapezoidalIntegrator(size_t dim, const AllocatorT& alloc = AllocatorT()) : tmp(dim, alloc) {}

        void prefix(matrix_span_t& A, double dt) {
            size_t len = A.length();

            tmp.copy_from(A[0]);
            A[0].zero();

            for (size_t i = 1; i < len; ++i) {
                A[i].scale(dt / 2).add(tmp, dt / 2).add(A[i - 1]);
                tmp.scale(-1).add(A[i], 2 / dt).add(A[i - 1], -2 / dt);
            }
        }

        void sum(matrix_span_t& A, matrix_t& out, double dt) {
            size_t len = A.length();

            out.add(A[0], dt / 2);
            for (size_t i = 1; i + 1 < len; ++i) out.add(A[i], dt);
            out.add(A[len - 1], dt / 2);
        }

        void prefix_vjp(matrix_span_t& A, double dt) {
            double half_dt = dt / 2;
            tmp.zero();
            for (size_t i = A.length() - 1; i > 0; --i) {
                A[i].scale(half_dt).add(tmp);
                tmp.scale(-1).add(A[i], 2);
            }
            A[0].copy_from(tmp);
            A[0].scale(0.5);
        }

        void sum_vjp(matrix_span_t& A, const matrix_t& out, double dt) {
            size_t len = A.length();
            A[0].copy_from(out);
            A[0].scale(dt / 2);
            for (size_t i = 1; i + 1 < len; ++i) {
                A[i].copy_from(out);
                A[i].scale(dt);
            }
            A[len - 1].copy_from(out);
            A[len - 1].scale(dt / 2);
        }

        void sum_vjp_add(matrix_span_t& A, const matrix_t& out, double dt) {
            size_t len = A.length();
            A[0].add(out, dt / 2);
            for (size_t i = 1; i + 1 < len; ++i) A[i].add(out, dt);
            A[len - 1].add(out, dt / 2);
        }

        matrix_t borrow_scratch() {
            return tmp.asView();
        }
    };


}

#endif
