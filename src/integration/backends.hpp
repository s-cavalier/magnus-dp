#ifndef __INTEGRATE_BACKENDS_HPP__
#define __INTEGRATE_BACKENDS_HPP__
#include "integrate.hpp"
#include "composer/dispatch.hpp"
#include "util/memorybuffer.hpp"

#include "default.hpp"
#include "trapezoid.hpp"
#include "simpson.hpp"
#include "boole.hpp"

namespace Magnus {

    struct DefaultIntBackend {
        template <class NumT, MatrixPolicy MatPolicy>
        using type = DefaultIntegrator<NumT, MatPolicy, MemoryBuffer::Allocator<NumT>>;

        static consteval std::string_view dispatch_name() { return "Riemann"; }

        static bool valid( [[maybe_unused]] const ParamsBase& ) { return true; }
        static bool use( [[maybe_unused]] const ParamsBase& ) { return true; }
    };

    struct TrapezoidBackend {
        template <class NumT, MatrixPolicy MatPolicy>
        using type = TrapezoidalIntegrator<NumT, MatPolicy, MemoryBuffer::Allocator<NumT>>;

        static consteval std::string_view dispatch_name() { return "Trapezoid"; }

        static bool valid( [[maybe_unused]] const ParamsBase& ) { return true; }
        static bool use( [[maybe_unused]] const ParamsBase& ) { return true; }
    };

    struct SimpsonBackend {
        template <class NumT, MatrixPolicy MatPolicy>
        using type = SimpsonIntegrator<NumT, MatPolicy, MemoryBuffer::Allocator<NumT>>;

        static consteval std::string_view dispatch_name() { return "Simpson"; }

        static bool valid( [[maybe_unused]] const ParamsBase& ) { return true; }
        static bool use( [[maybe_unused]] const ParamsBase& ) { return true; }
    };

    struct BooleBackend {
        template <class NumT, MatrixPolicy MatPolicy>
        using type = BooleIntegrator<NumT, MatPolicy, MemoryBuffer::Allocator<NumT>>;

        static consteval std::string_view dispatch_name() { return "Boole"; }

        static bool valid( [[maybe_unused]] const ParamsBase& ) { return true; }
        static bool use( [[maybe_unused]] const ParamsBase& ) { return true; }
    };

    using IntegratorBackends = type_list<
        BooleBackend,
        SimpsonBackend,
        TrapezoidBackend,
        DefaultIntBackend
    >;

}

#endif
