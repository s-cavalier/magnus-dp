#ifndef __GAUSS_LEGENDRE_BACKENDS_HPP__
#define __GAUSS_LEGENDRE_BACKENDS_HPP__

#include "composer/dispatch.hpp"
#include "gausslegendre.hpp"

namespace Magnus {

    namespace detail {

        inline bool enough_parallel_gl_work(const ParamsBase& p) {
            if (p.n <= 1) return false;

            constexpr size_t min_parallel_work = 30'000;
            const size_t gl_order = (p.n + 3) / 2;
            const size_t work_per_sample = (p.n - 1) * gl_order;
            const size_t min_samples = (min_parallel_work + work_per_sample - 1) / work_per_sample;
            return p.samples >= min_samples;
        }

    }

    struct GLThreadPoolBackend {
        using type = GL_threadpool;

        static constexpr std::string_view dispatch_name() { return "threadpool"; }
        static bool valid([[maybe_unused]] const ParamsBase&) {
            return detail::has_gl_thread_pool();
        }

        static bool use(const ParamsBase& p) {
            return valid(p) && detail::enough_parallel_gl_work(p);
        }
    };

    struct GLForLoopBackend {
        using type = GL_forloop;

        static constexpr std::string_view dispatch_name() { return "serial"; }
        static bool valid([[maybe_unused]] const ParamsBase&) { return true; }
        static bool use([[maybe_unused]] const ParamsBase&) { return true; }
    };

    struct GLOpenMPBackend {
        using type = GL_openmp;

        static constexpr std::string_view dispatch_name() { return "openmp"; }
        static bool valid([[maybe_unused]] const ParamsBase&) { return true; }

        static bool use(const ParamsBase& p) {
            return valid(p) && detail::enough_parallel_gl_work(p);
        }
    };

    using GLBackends = type_list<
        GLThreadPoolBackend,
        GLOpenMPBackend,
        GLForLoopBackend
    >;

}

#endif
