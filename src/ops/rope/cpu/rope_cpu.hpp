#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_size, size_t num_heads, size_t head_size, float theta);
} // namespace llaisys::ops::cpu