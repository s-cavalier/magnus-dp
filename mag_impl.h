#ifndef __MAG_IMPL_H__
#define __MAG_IMPL_H__
#include <iostream>
#include <array>
#include <vector>
#include <cuda/mdspan>
#include "interp.h"

template <size_t Degree>
void magnus_impl(size_t samples, float dt, float* in, float* out) {
    float4* data_d;
    cudaMalloc((void**)&data_d, sizeof(float4) * samples);

    std::vector<float4> packed(samples);
    for (size_t i = 0; i < samples; ++i) packed[i] = make_float4( in[3*i], in[3*i + 1], in[3*i + 2], 0.0f );
    cudaMemcpy( data_d, packed.data(), sizeof(float4) * samples );

    DeviceLinearInterp3 lerp( std::span<float4>( data_d, samples ), dt );

}




#endif