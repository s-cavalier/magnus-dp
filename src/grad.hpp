#ifndef __GRAD_HPP__
#define __GRAD_HPP__
#include "graddata.hpp"
#include "magnus.hpp"
#include "integrate.hpp"

namespace Magnus::VJP {

    template <Integrator Int>
    void one(
        typename Int::matrix_span_t& dA,
        typename Int::matrix_t& dOut,
        size_t n,
        typename Int::matrix_span_t& A,
        double t0,
        double tf,
        const VJP::OneData<typename Int::numeric_t>* one_data,
        const typename Int::allocator_t& alloc = typename Int::allocator_t()
    ) {
        using NumT = typename Int::numeric_t;
        using MatrixT = typename Int::matrix_t;
        using SpanT = typename Int::matrix_span_t;
        using MatPolicyT = typename Int::matrix_policy_t;
        
        size_t dim = A.mat_dim();
        size_t samples = A.length();
        size_t matrix_size = dim * dim;
        size_t span_size = samples * matrix_size;
        double dt = (tf - t0) / (samples - 1);

        // replace later with a matrix policy impl
        std::fill_n(dA.data(), dA.size(), NumT{0});

        Int integrator(dim, alloc);

        if (n == 1) {
            integrator.sum_vjp(dA, dOut, dt);
            return;
        };

        auto gl_table = GLTable::get();
        GLTable::DataView view = gl_table->get_order((n + 3) / 2);

        // replace with unique ptrs probably
        NumT* dY_data = alloc.allocate(span_size);
        NumT* dP_data = alloc.allocate(span_size);
        NumT* dTotal_data = alloc.allocate(matrix_size);
        NumT* B_data = alloc.allocate(matrix_size);

        SpanT dY(dY_data, dim, samples);
        SpanT dP(dP_data, dim, samples);
        MatrixT dTotal(dTotal_data, dim);
        MatrixT B(B_data, dim);


        MatrixT tmp = integrator.borrow_scratch();

    }

    

}


#endif