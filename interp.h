#ifndef __INTERP_H__
#define __INTERP_H__
#include <span>
#include <cuda/mdspan>

struct DeviceLinearInterp3 {
    std::span<float4> samples;
    float dt;
    float inv_dt;
    
    __host__ __device__
    DeviceLinearInterp3( std::span<float4> samples, float dt ) : samples(samples), dt(dt), inv_dt(1 / dt) {}

    __device__ inline float3 eval(float t) const {
        float u = t * inv_dt;

        int i = (int)u;

        if (i < 0) i = 0;
        if (i > samples.size() - 2) i = samples.size() - 2;

        float frac = u - i;

        float4 a = samples[i];
        float4 b = samples[i + 1];

        return make_float3(
            a.x + frac * (b.x - a.x),
            a.y + frac * (b.y - a.y),
            a.z + frac * (b.z - a.z)
        );

    }

};

#endif