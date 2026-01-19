#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
    void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
               llaisysDataType_t dtype,
               size_t seq_len, size_t intermediate_dim);
               // out: [seq_len, intermediate_dim]
               // gate: [seq_len, intermediate_dim]
               // up: [seq_len, intermediate_dim]
} // namespace llaisys::ops::cpu