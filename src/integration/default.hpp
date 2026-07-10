#ifndef __INTEGRATION_DEFAULT_HPP__
#define __INTEGRATION_DEFAULT_HPP__
#include "integrate.hpp"

namespace Magnus {

    template <class NumT, MatrixPolicy MatPolicyT, class AllocatorT = std::allocator<NumT>>
    class DefaultIntegrator {
        DynMatrix<NumT, MatPolicyT, AllocatorT> tmp;

    public:
        using numeric_t = NumT;
        using allocator_t = AllocatorT;
        using matrix_policy_t = MatPolicyT;
        using matrix_t = MatrixView<NumT, MatPolicyT>;
        using matrix_span_t = MatrixSpan<NumT, MatPolicyT>;
        
        static constexpr size_t divisibility_requirement() { return 1; }
        static constexpr size_t memory_requirement() { return 1; }

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


}

#endif