#ifndef __BACKEND_COMPOSER_HPP__
#define __BACKEND_COMPOSER_HPP__
#include "dispatch.hpp"
#include "integratebackends.hpp"
#include "matrixbackends.hpp"

namespace Magnus {

    std::unique_ptr<KernelPlan> make_plan(Params& p, size_t num_idx, size_t mat_idx, size_t int_idx);

}

#endif