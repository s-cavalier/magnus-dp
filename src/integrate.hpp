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

        { T::divisibility_requirement() } -> std::same_as<size_t>;
        { T::memory_requirement() } -> std::same_as<size_t>;

        { T( size_t{}, alloc ) };
        { integrator.prefix( data, double{} ) } -> std::same_as<void>;
        { integrator.sum( data, out, double{} ) } -> std::same_as<void>;
        { integrator.borrow_scratch() } -> std::same_as< MatrixView<typename T::numeric_t, typename T::matrix_policy_t> >;
    };




}

#endif
