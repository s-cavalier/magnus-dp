#ifndef __MAG_DISPATCH_H__
#define __MAG_DISPATCH_H__
#include "mag_impl.h"

#ifndef MAX_DEGREE
    #error "MAX_DEGREE is not defined. Please define it to set the flag for the degree of the magnus compilation."
#endif

#if MAX_DEGREE < 1
    #error "MAX_DEGREE should be at least 1 or higher."
#endif

#ifndef MIN_DEGREE
#define MIN_DEGREE 1
#endif

#if MIN_DEGREE < 1
    #error "MIN_DEGREE should be at least 1 or higher."
#endif

#if MAX_DEGREE < MIN_DEGREE
    #error "Max degree must be greater than or equal to min degree"
#endif

static_assert( MAX_DEGREE > 0 );

static constexpr size_t NO_DEGREE = MAX_DEGREE - MIN_DEGREE + 1;

using magnus_impl_t = void(*)(size_t, float, float*, float*);

template <size_t... Degrees>
void magnus_dispatch(size_t degree, size_t samples, float dt, float* in, float* out, std::index_sequence<Degrees...> seq) {
    static constexpr std::array<magnus_impl_t, sizeof...(Degrees)> dispatcher{ magnus_impl<Degrees + MIN_DEGREE> ... };

    dispatcher[degree - MIN_DEGREE](samples, dt, in, out);
}

template <size_t Degree>
void magnus_dispatch_helper(size_t degree, size_t samples, float dt, float* in, float* out) {
    magnus_dispatch(degree, samples, dt, in, out, std::make_index_sequence<Degree>());
}

extern "C" void magnus(size_t degree, size_t samples, float dt, float* in, float* out) {
    magnus_dispatch_helper<NO_DEGREE>(degree, samples, dt, in, out);
}

#endif