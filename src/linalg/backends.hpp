#ifndef __MAT_BACKENDS_HPP__
#define __MAT_BACKENDS_HPP__
#include "composer/dispatch.hpp"
#include "fixed.hpp"
#include "fixed_config.hpp"
#include "manual.hpp"
#include "blas.hpp"
#include "spacecurve.hpp"

namespace Magnus {

    template <size_t N>
    struct FixedBackend {
        template <Numeric NumT>
        using type = FixedDimPolicy<NumT, N>;

        static inline constexpr auto dispatch_name_storage = make_integral_name<N>("Fixed", "");

        static constexpr std::string_view dispatch_name() {
            return {
                dispatch_name_storage.data(),
                dispatch_name_storage.size()
            };
        }

        static bool valid(const ParamsBase& p) { return p.dim == N; }
        static bool use(const ParamsBase& p) { return valid(p); }
    };

    struct ManualBackend {
        template <Numeric NumT>
        using type = ManualPolicy<NumT>;

        static constexpr std::string_view dispatch_name() { return "Manual"; }

        static bool valid([[maybe_unused]] const ParamsBase&) { return true; }
        static bool use(const ParamsBase& p) { return p.dim < 64; }
    };

    struct BlasBackend {
        template <Numeric NumT>
        using type = BlasPolicy<NumT>;

        static constexpr std::string_view dispatch_name() { return "Blas"; }

        static bool valid([[maybe_unused]] const ParamsBase&) { return true; }
        static bool use([[maybe_unused]] const ParamsBase&) { return true; }
    };

namespace SpaceCurve {

    struct Backend {
        template <Numeric NumT>
        using type = SpaceCurve::Policy<NumT>;

        static constexpr std::string_view dispatch_name() { return "SpaceCurve"; }

        static bool valid(const ParamsBase& p) { return p.dim == 2; }
        static bool use([[maybe_unused]] const ParamsBase&) { return true; }
    };

}

    using MatrixBackends = type_list<
#if MAGNUS_FIXED_ENABLE_2
        FixedBackend<2>,
#endif
#if MAGNUS_FIXED_ENABLE_3
        FixedBackend<3>,
#endif
#if MAGNUS_FIXED_ENABLE_4
        FixedBackend<4>,
#endif
#if MAGNUS_FIXED_ENABLE_5
        FixedBackend<5>,
#endif
#if MAGNUS_FIXED_ENABLE_6
        FixedBackend<6>,
#endif
#if MAGNUS_FIXED_ENABLE_7
        FixedBackend<7>,
#endif
#if MAGNUS_FIXED_ENABLE_8
        FixedBackend<8>,
#endif
        ManualBackend,
        BlasBackend,
        SpaceCurve::Backend
    >;

}

#endif
