#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype,
            size_t batch_size, size_t in_features, size_t out_features);
            // in : [batch_size, in_features]
            // weight: [out_features, in_features]
            // out: [batch_size, out_features]
            // bias: [out_features]
} // namespace llaisys::ops::cpu

