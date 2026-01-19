#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_id,
           const int64_t seq_len,
           const int64_t num_heads,
           const int64_t head_dim, float theta = 10000.0f) {
    std::vector<float> angles;
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < head_dim / 2; ++j) {
            if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                float pos = llaisys::utils::cast<float>(pos_id[i]);
                auto angle = pos / std::pow(theta, (2.0f * j) / head_dim);
                angles.push_back(angle);
            } else {
                auto angle = static_cast<float>(pos_id[i]) / std::pow(theta, (2.0f * j) / head_dim);
                angles.push_back(angle);
            }
        }
    }

    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < num_heads; ++j) {
            for (int64_t k = 0; k < head_dim / 2; ++k) {
                auto a = in[i * num_heads * head_dim + j * head_dim + k];
                auto b = in[i * num_heads * head_dim + j * head_dim + k + head_dim / 2];
                if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    float a_f = llaisys::utils::cast<float>(a);
                    float b_f = llaisys::utils::cast<float>(b);
                    out[i * num_heads * head_dim + j * head_dim + k] = llaisys::utils::cast<T>(a_f * std::cos(angles[i * head_dim / 2 + k]) - b_f * std::sin(angles[i * head_dim / 2 + k]));
                    out[i * num_heads * head_dim + j * head_dim + k + head_dim / 2] = llaisys::utils::cast<T>(b_f * std::cos(angles[i * head_dim / 2 + k]) + a_f * std::sin(angles[i * head_dim / 2 + k]));
                } else {
                    out[i * num_heads * head_dim + j * head_dim + k] = a * std::cos(angles[i * head_dim / 2 + k]) - b * std::sin(angles[i * head_dim / 2 + k]);
                    out[i * num_heads * head_dim + j * head_dim + k + head_dim / 2] = b * std::cos(angles[i * head_dim / 2 + k]) + a * std::sin(angles[i * head_dim / 2 + k]);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_size, size_t num_heads, size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_size, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_size, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids),
                     seq_size, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu