#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <cstddef>
#include <memory>

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

}

#endif
