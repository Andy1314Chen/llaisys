#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t seq_len, size_t total_seq_len, size_t head_dim, size_t head_dim_v,
                    size_t num_heads, size_t num_kv_heads, float scale);
// out: [seq_len, num_heads, head_dim_v]
// q : [seq_len, num_heads, head_dim]
// k : [total_len, nkvhead, head_dim]
// v : [total_len, nkvhead, head_dim_v]
// scale: float scaling factor for qk^T
} // namespace llaisys::ops::cpu