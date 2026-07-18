#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <cstddef>
#include <memory>
#include "integration/integrate.hpp"
#include "mem/memorybuffer.hpp"

namespace Magnus {

    constexpr size_t gl_max_n(size_t n) {
        return (n + 3) / 2;
    }

    constexpr size_t total_orders(size_t n) {
        return n - 1;
    }

    template <class AllocatorT>
    class AllocatorDeleter {
        [[no_unique_address]] AllocatorT m_alloc;
        size_t m_count;

    public:
        using pointer = typename std::allocator_traits<AllocatorT>::pointer;

        AllocatorDeleter(const AllocatorT& alloc, size_t count) :
            m_alloc(alloc),
            m_count(count) {}

        void operator()(pointer ptr) {
            if (ptr != nullptr) std::allocator_traits<AllocatorT>::deallocate(m_alloc, ptr, m_count);
        }

        const AllocatorT& allocator() const { return m_alloc; }
        size_t count() const { return m_count; }
    };

    template <class T, class AllocatorT = std::allocator<T>>
    auto allocate_unique(size_t count, AllocatorT alloc = AllocatorT()) {
        using traits_t = std::allocator_traits<AllocatorT>;
        using value_t = typename traits_t::value_type;

        typename traits_t::pointer ptr = traits_t::allocate(alloc, count);
        return std::unique_ptr<T, AllocatorDeleter<AllocatorT>>(
            ptr,
            AllocatorDeleter<AllocatorT>(alloc, count)
        );
    }

    template <Integrator Int>
    struct alignas(CACHE_LINE_ALIGNMENT) FwdPathWorkspace {
        using NumT = typename Int::numeric_t;
        using PolicyT = typename Int::matrix_policy_t;
        using MatrixT = typename Int::matrix_t;
        using AllocT = typename Int::allocator_t;

        Int integrator;
        MatrixT temp;
        DynMatrixSpan<NumT, PolicyT, AllocT> Y;
        DynMatrix<NumT, PolicyT, AllocT> total_copy;

        FwdPathWorkspace(size_t dim, size_t samples, const AllocT& alloc) :
            integrator(dim, alloc),
            temp(integrator.borrow_scratch()),
            Y(dim, samples, alloc),
            total_copy(dim, alloc) {}
    };

    template <Integrator Int>
    struct alignas(CACHE_LINE_ALIGNMENT) FwdWorkspace : FwdPathWorkspace<Int> {
        using BaseT = FwdPathWorkspace<Int>;
        using NumT = typename BaseT::NumT;
        using MatrixSpanT = typename Int::matrix_span_t;
        using AllocT = typename BaseT::AllocT;

        MatrixSpanT partial_out;

        FwdWorkspace(size_t dim, size_t samples, size_t output_count, NumT* p_out, const AllocT& alloc ) :
            BaseT(dim, samples, alloc),
            partial_out(p_out, dim, output_count)
        {
            partial_out.zero();
        }

    };

    template <Integrator Int, size_t Alignment = CACHE_LINE_ALIGNMENT>
    constexpr size_t fwd_workspace_buffer_bytes(
        size_t worker_count,
        size_t dim,
        size_t output_count
    ) {
        if (worker_count <= 1) return 0;

        using NumT = typename Int::numeric_t;
        size_t output_bytes = dim * dim * output_count * sizeof(NumT);

        return (worker_count - 1) * (output_bytes + Alignment - 1);
    }

}

#endif
