#ifndef __INTEGRATE_HPP__
#define __INTEGRATE_HPP__
#include "matrix.hpp"

namespace Magnus {

    template <class T>
    concept Integrator = requires(
        T& integrator,
        MatrixSpan<typename T::numeric_t, typename T::matrix_policy_t>& data,
        MatrixView<typename T::numeric_t, typename T::matrix_policy_t>& out,
        const typename T::allocator_t& alloc
    ) {
        typename T::allocator_t;
        typename T::numeric_t;
        typename T::matrix_policy_t;
        typename T::matrix_t;
        typename T::matrix_span_t;
        requires std::same_as< typename T::matrix_t, MatrixView<typename T::numeric_t, typename T::matrix_policy_t> >;
        requires std::same_as< typename T::matrix_span_t, MatrixSpan<typename T::numeric_t, typename T::matrix_policy_t> >;
        requires std::same_as<typename T::matrix_policy_t::numeric_t, typename T::numeric_t>;

        { T( size_t{}, alloc ) };
        { integrator.prefix( data, double{} ) } -> std::same_as<void>;
        { integrator.sum( data, out, double{} ) } -> std::same_as<void>;
        { integrator.borrow_scratch() } -> std::same_as< MatrixView<typename T::numeric_t, typename T::matrix_policy_t> >;
    };

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class DefaultIntegrator {
        DynMatrix<NumT, MatPolicyT, AllocatorT> tmp;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

        DefaultIntegrator( size_t dim, const AllocatorT& alloc = AllocatorT() ) : tmp(dim, alloc) {}

        void prefix(matrix_span_t& A, double dt) {
            A[0].scale(dt);
            for (size_t i = 1; i < A.length(); ++i) A[i].scale(dt).add(A[i - 1]);
        }

        void sum(matrix_span_t& A, matrix_t& out, double dt) {
            for (size_t i = 0; i < A.length(); ++i) out.add( A[i], dt );
        }

        matrix_t borrow_scratch() {
            return tmp.asView();
        }

    };

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class TrapezoidalIntegrator {
        DynMatrix<NumT, MatPolicyT, AllocatorT> tmp;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

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

        matrix_t borrow_scratch() {
            return tmp.asView();
        }
    };

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class SimpsonIntegrator {
        DynMatrixSpan<NumT, MatPolicyT, AllocatorT> scratch;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

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

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class BooleIntegrator {
        DynMatrixSpan<NumT, MatPolicyT, AllocatorT> scratch;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;

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
