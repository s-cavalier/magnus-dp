#ifndef __GAUSS_LEGENDRE_BACKENDS_HPP__
#define __GAUSS_LEGENDRE_BACKENDS_HPP__

#include "composer/dispatch.hpp"
#include "gausslegendre.hpp"

namespace Magnus {

    struct GLForLoopBackend {
        using type = GL_forloop;

        static constexpr std::string_view dispatch_name() { return "serial"; }
        static bool valid([[maybe_unused]] const ParamsBase&) { return true; }
        static bool use([[maybe_unused]] const ParamsBase&) { return true; }
    };

    struct GLOpenMPBackend {
        using type = GL_openmp;
        static constexpr size_t min_parallel_n = 8;

        static constexpr std::string_view dispatch_name() { return "openmp"; }
        static bool valid([[maybe_unused]] const ParamsBase&) { return true; }

        static bool use(const ParamsBase& p) {
            return valid(p) && p.n >= min_parallel_n;
        }
    };

    using GLBackends = type_list<
        GLOpenMPBackend,
        GLForLoopBackend
    >;

}

#endif
