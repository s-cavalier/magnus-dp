#ifndef __GAUSS_LEGENDRE_HPP__
#define __GAUSS_LEGENDRE_HPP__
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <memory>
#include <atomic>
#include <concepts>
#include <algorithm>
#include <exception>
#include <functional>
#include <latch>
#include <mutex>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Magnus {

    namespace detail {

        class GLThreadPool {
        public:
            virtual ~GLThreadPool() = default;

            virtual size_t thread_count() const = 0;
            virtual void schedule(std::move_only_function<void()> fn) = 0;
        };

        inline thread_local GLThreadPool* current_gl_thread_pool = nullptr;

        class ScopedGLThreadPool {
            GLThreadPool* previous;

        public:
            explicit ScopedGLThreadPool(GLThreadPool& pool) noexcept :
                previous(std::exchange(current_gl_thread_pool, &pool)) {}

            ~ScopedGLThreadPool() {
                current_gl_thread_pool = previous;
            }

            ScopedGLThreadPool(const ScopedGLThreadPool&) = delete;
            ScopedGLThreadPool& operator=(const ScopedGLThreadPool&) = delete;
            ScopedGLThreadPool(ScopedGLThreadPool&&) = delete;
            ScopedGLThreadPool& operator=(ScopedGLThreadPool&&) = delete;
        };

        inline bool has_gl_thread_pool() noexcept {
            return current_gl_thread_pool != nullptr;
        }

    }

    /*
    An abstract handle onto some GaussLegendre table.
    We use a global PIMPL approach so that way the details of "where" the data is stored is abstracted away.
    It could be a statically linked GL table, mmapped table, heap table, or even just a file.

    Assumes doubles. Maybe extend later for other purposes.
    */
    class GLTable {
        static std::atomic<std::shared_ptr<const GLTable>> global_table;

    protected:
        size_t m_max_order;
        GLTable(size_t max_order);

    public:
        class DataView {
            const double* m_w;
            const double* m_x;
            size_t n;

        public:
            DataView(const double* weights, const double* nodes, size_t order);

            size_t order() const;
            std::pair<double, double> operator[](size_t i) const;
            std::pair<double, double> at(size_t i) const;

            const double* weights() const;
            const double* nodes() const;
        };

        virtual ~GLTable();
        virtual DataView get_order(size_t n) const = 0;

        size_t max_order() const;

        static std::shared_ptr<const GLTable> get();
        static void update(std::shared_ptr<const GLTable> new_table);

    };

    class StaticTable : public GLTable {
        struct PyGLTableHeader {
            char magic[4];
            uint32_t max_n;

            size_t offsets_offset;
            size_t nodes_offset;
            size_t weights_offset;
        };

        size_t* offsets;
        double* weights;
        double* nodes;

    public:
        StaticTable(std::byte* table);

        DataView get_order(size_t n) const override;

    };

    class OwningTable final : public GLTable {
        std::vector<size_t> offsets;
        std::vector<double> weights;
        std::vector<double> nodes;

    public:
        OwningTable(
            size_t max_order,
            std::vector<size_t> offsets,
            std::vector<double> weights,
            std::vector<double> nodes
        );

        DataView get_order(size_t n) const override;
    };

    struct GL_forloop {
        static constexpr size_t lane_count([[maybe_unused]]size_t) { return 1; }

        static void invoke(size_t order, auto&& fn) {
            for (size_t q = 0; q < order; ++q) std::invoke(fn, q, 0);
        }

    };

    struct GL_openmp {
        static size_t lane_count(size_t order) {
#ifdef _OPENMP
            return std::min(order, size_t(omp_get_max_threads()));
#else
            return std::min(order, size_t{1});
#endif
        }

        static void invoke(size_t order, auto&& fn) {
#ifdef _OPENMP
            int lanes = static_cast<int>(lane_count(order));
            #pragma omp parallel for schedule(static) num_threads(lanes)
            for (size_t q = 0; q < order; ++q) {
                std::invoke(fn, q, omp_get_thread_num());
            }
#else
            for (size_t q = 0; q < order; ++q) std::invoke(fn, q, 0);
#endif
        }

    };

    struct GL_threadpool {
        static size_t lane_count(size_t order) {
            if (!detail::has_gl_thread_pool()) {
                throw std::runtime_error("no thread pool is active for the current call");
            }

            const size_t threads = std::max(size_t{1}, detail::current_gl_thread_pool->thread_count());
            return std::min(order, threads);
        }

        static void invoke(size_t order, auto&& fn) {
            detail::GLThreadPool* pool = detail::current_gl_thread_pool;
            if (pool == nullptr) {
                throw std::runtime_error("no thread pool is active for the current call");
            }

            const size_t lanes = lane_count(order);
            std::latch done(static_cast<std::ptrdiff_t>(lanes - 1));
            std::mutex error_mutex;
            std::exception_ptr first_error;

            auto record_error = [&](std::exception_ptr error) {
                std::lock_guard lock(error_mutex);
                if (!first_error) first_error = std::move(error);
            };

            auto run_lane = [&](size_t lane) {
                try {
                    const size_t begin = order * lane / lanes;
                    const size_t end = order * (lane + 1) / lanes;
                    for (size_t q = begin; q < end; ++q) std::invoke(fn, q, lane);
                } catch (...) {
                    record_error(std::current_exception());
                }
            };

            for (size_t lane = 1; lane < lanes; ++lane) {
                try {
                    pool->schedule([&, lane] {
                        run_lane(lane);
                        done.count_down();
                    });
                } catch (...) {
                    record_error(std::current_exception());
                    done.count_down();
                }
            }

            run_lane(0);
            done.wait();

            if (first_error) std::rethrow_exception(first_error);
        }
    };

}


#endif
