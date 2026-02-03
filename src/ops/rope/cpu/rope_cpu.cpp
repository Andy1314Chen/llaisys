#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_id,
           const int64_t seq_len,
           const int64_t num_heads,
           const int64_t head_dim, double theta = 10000.0) {
    // 预计算 sin/cos，用 double 精度计算 angle 以匹配 PyTorch
    std::vector<float> sin_cache(seq_len * head_dim / 2);
    std::vector<float> cos_cache(seq_len * head_dim / 2);

    for (int64_t i = 0; i < seq_len; ++i) {
        double pos = static_cast<double>(pos_id[i]);
        for (int64_t j = 0; j < head_dim / 2; ++j) {
            // 用 double 精度计算 angle，与 Python 的 ** 运算符行为一致
            double angle = pos / std::pow(theta, (2.0 * j) / head_dim);
            size_t idx = i * head_dim / 2 + j;
            sin_cache[idx] = static_cast<float>(std::sin(angle));
            cos_cache[idx] = static_cast<float>(std::cos(angle));
        }
    }

    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < num_heads; ++j) {
            for (int64_t k = 0; k < head_dim / 2; ++k) {
                size_t base = i * num_heads * head_dim + j * head_dim;
                size_t cache_idx = i * head_dim / 2 + k;

                float a = llaisys::utils::cast<float>(in[base + k]);
                float b = llaisys::utils::cast<float>(in[base + k + head_dim / 2]);
                float s = sin_cache[cache_idx];
                float c = cos_cache[cache_idx];

                out[base + k] = llaisys::utils::cast<T>(a * c - b * s);
                out[base + k + head_dim / 2] = llaisys::utils::cast<T>(b * c + a * s);
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