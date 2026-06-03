#ifndef __DISPATCH_HPP__
#define __DISPATCH_HPP__
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <tuple>
#include <stdexcept>
#include <variant>
#include "memorybuffer.hpp"
#include "integrate.hpp"
#include "magnus.hpp"

namespace Magnus {

    template <class NumT>
    concept valid_num_t = 
        std::same_as<NumT, float> || 
        std::same_as<NumT, double> || 
        std::same_as<NumT, std::complex<float>> || 
        std::same_as<NumT, std::complex<double>>;

    struct Parameters {
        template <class NumT>
        using alloc_t = MemoryBuffer::Allocator<NumT, 8>;
        using f32 = float;
        using f64 = double;
        using c32 = std::complex<float>;
        using c64 = std::complex<double>;

        template <valid_num_t NumT>
        struct typed_data {
            NumT* in;
            NumT* out;
            alloc_t<NumT> alloc;
        };

        std::variant<typed_data<f32>, typed_data<f64>, typed_data<c32>, typed_data<c64>> data;
        size_t n;
        size_t dim;
        size_t samples;
        f64 t0; f64 tf;

        template <valid_num_t NumT>
        Parameters( NumT* in, NumT* out, const alloc_t<NumT>& alloc, size_t n, size_t dim, size_t samples, f64 t0, f64 tf ) :
            data( std::in_place_type<typed_data<NumT>>, in, out, alloc ),
            n(n), dim(dim),
            samples(samples),
            t0(t0), tf(tf)
        {}

        template <valid_num_t NumT>
        NumT* in_data() {
            return std::get<typed_data<NumT>>( data ).in;
        }

        template <valid_num_t NumT>
        NumT* out_data() {
            return std::get<typed_data<NumT>>( data ).out;
        }

        template <valid_num_t NumT>
        const alloc_t<NumT>& get_allocator() const {
            return std::get<typed_data<NumT>>( data ).alloc;
        }

        template<class Visitor>
        decltype(auto) visit(Visitor&& visitor) & {
            return std::visit(std::forward<Visitor>(visitor), data);
        }

        template<class Visitor>
        decltype(auto) visit(Visitor&& visitor) const& {
            return std::visit(std::forward<Visitor>(visitor), data);
        }

        template<class Visitor>
        decltype(auto) visit(Visitor&& visitor) && {
            return std::visit(std::forward<Visitor>(visitor), std::move(data));
        }

        template<class Visitor>
        decltype(auto) visit(Visitor&& visitor) const&& {
            return std::visit(std::forward<Visitor>(visitor), std::move(data));
        }

    };


}

#endif