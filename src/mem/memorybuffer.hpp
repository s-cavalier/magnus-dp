#ifndef __MEMORY_BUFFER_HPP__
#define __MEMORY_BUFFER_HPP__

#include <cstddef>
#include <functional>
#include <new>
#include <span>
#include <cstdlib>
#include <memory>

namespace Magnus {

    static constexpr size_t DEFAULT_ALIGNMENT = 32;
    static constexpr size_t CACHE_LINE_ALIGNMENT = std::hardware_constructive_interference_size;

    class MemoryBuffer {
        std::byte* buf;
        size_t capacity;
        size_t offset;
        bool owns;

        template <class T, size_t Alignment = DEFAULT_ALIGNMENT> requires ( Alignment >= DEFAULT_ALIGNMENT && ( ( Alignment & (Alignment - 1) ) == 0 ) )
        T* alloc(size_t num) {
            if (num == 0) throw std::bad_alloc();

            size_t bytes = sizeof(T) * num;
            std::byte* current = buf + offset;
            size_t space = capacity - offset;
            void* aligned_ptr = current;

            if ( !std::align( Alignment, bytes, aligned_ptr, space ) ) throw std::bad_alloc();

            std::byte* aligned_byte_ptr = static_cast<std::byte*>(aligned_ptr);
            offset = ( aligned_byte_ptr - buf ) + bytes;

            return reinterpret_cast<T*>(aligned_ptr);
        }

    public:
        MemoryBuffer(size_t capacity)
            : buf( new std::byte[capacity] ),
            capacity(capacity),
            offset(0),
            owns(true) {}

        MemoryBuffer(std::span<std::byte> ext, bool owns = false)
            : buf(ext.data()), capacity(ext.size()), offset(0), owns(owns) {}

        ~MemoryBuffer() {
            if (owns) delete[] buf;
        }

        void reset() {
            offset = 0;
        }

        MemoryBuffer(const MemoryBuffer&) = delete;
        MemoryBuffer& operator=(const MemoryBuffer&) = delete;
        MemoryBuffer(MemoryBuffer&& other) = delete;
        MemoryBuffer& operator=(MemoryBuffer&& other) = delete;

        template <class T, size_t Alignment = DEFAULT_ALIGNMENT>
        class Allocator {
            friend class MemoryBuffer;

            std::reference_wrapper<MemoryBuffer> buffer;

            Allocator(MemoryBuffer& buf) : buffer(buf) {}

        public:
            using value_type = T;
            using propagate_on_container_copy_assignment = std::true_type;
            using propagate_on_container_move_assignment = std::true_type;
            using propagate_on_container_swap = std::true_type;
            using is_always_equal = std::false_type;

            template <class U, size_t UAlignment = Alignment>
            Allocator(const Allocator<U, UAlignment>& other) : buffer(other.buffer.get()) {}
            template<class U, size_t UAlignment = Alignment> struct rebind { using other = Allocator<U, UAlignment>; };

            T* allocate(size_t n) {
                return buffer.get().template alloc<T, Alignment>(n);
            }

            void deallocate([[maybe_unused]] T* p, [[maybe_unused]] size_t n) {}  // no-op for bump allocator

            bool operator==(const Allocator& other) const {
                return &buffer.get() == &other.buffer.get();
            }

            bool operator!=(const Allocator& other) const {
                return !(*this == other);
            }
        };

        template <class T, size_t Alignment = DEFAULT_ALIGNMENT>
        Allocator<T, Alignment> get_allocator() {
            return Allocator<T, Alignment>(*this);
        }
    };

}

#endif
