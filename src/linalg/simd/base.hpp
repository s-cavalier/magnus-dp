#ifndef __LINALG_SIMD_BASE_HPP__
#define __LINALG_SIMD_BASE_HPP__

#include <concepts>
#include <string_view>

namespace Magnus::SIMD {

template <class T>
concept IBackend = requires 
{
    typename T::arch_t;
    { T::available() } -> std::same_as<bool>;
    { T::name() } -> std::same_as<std::string_view>;
};


}  // namespace Magnus::SIMD

#endif