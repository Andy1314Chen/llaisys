#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t seq_len, size_t total_seq_len, size_t head_dim, size_t head_dim_v,
                     size_t num_heads, size_t kv_num_heads, float scale) {
    const size_t groups = num_heads / kv_num_heads;
    const size_t seq_offset = total_seq_len - seq_len;

    // Buffers to avoid repeated allocation
    std::vector<float> scores(total_seq_len);
    std::vector<float> out_accum(head_dim_v);

    for (size_t h = 0; h < num_heads; ++h) {
        size_t kv_h = h / groups;

        for (size_t i = 0; i < seq_len; ++i) {
            // Absolute position of the current query in the total sequence
            size_t pos = seq_offset + i;

            // Pointer to current Query vector: [seq_len, num_heads, head_dim]
            const T *q_vec = q + (i * num_heads + h) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();

            // 1. Calculate Attention Scores
            for (size_t j = 0; j < total_seq_len; ++j) {
                // Causal Masking: only attend to past and current tokens
                if (j > pos) {
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // Pointer to Key vector: [total_seq_len, kv_num_heads, head_dim]
                const T *k_vec = k + (j * kv_num_heads + kv_h) * head_dim;

                float score = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                        score += static_cast<float>(llaisys::utils::cast<float>(q_vec[d]) * llaisys::utils::cast<float>(k_vec[d]));
                    } else {
                        score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
                    }
                }
                score *= scale;
                scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_seq_len; ++j) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    scores[j] = 0.0f;
                } else {
                    float val = std::exp(scores[j] - max_score);
                    scores[j] = val;
                    sum_exp += val;
                }
            }

            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            // 3. Weighted Sum of Values
            std::fill(out_accum.begin(), out_accum.end(), 0.0f);

            for (size_t j = 0; j < total_seq_len; ++j) {
                float weight = scores[j] * inv_sum;
                if (weight == 0.0f) {
                    continue;
                }

                // Pointer to Value vector: [total_seq_len, kv_num_heads, head_dim_v]
                const T *v_vec = v + (j * kv_num_heads + kv_h) * head_dim_v;

                for (size_t d = 0; d < head_dim_v; ++d) {
                    if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                        out_accum[d] += weight * static_cast<float>(llaisys::utils::cast<float>(v_vec[d]));
                    } else {
                        out_accum[d] += weight * static_cast<float>(v_vec[d]);
                    }
                }
            }

            // 4. Store Output: [seq_len, num_heads, head_dim_v]
            T *out_vec = out + (i * num_heads + h) * head_dim_v;
            for (size_t d = 0; d < head_dim_v; ++d) {
                if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                    out_vec[d] = llaisys::utils::cast<T>(out_accum[d]);
                } else {
                    out_vec[d] = static_cast<T>(out_accum[d]);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t seq_len, size_t total_seq_len, size_t head_dim, size_t head_dim_v,
                    size_t num_heads, size_t num_kv_heads, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               seq_len, total_seq_len, head_dim, head_dim_v, num_heads, num_kv_heads, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               seq_len, total_seq_len, head_dim, head_dim_v, num_heads, num_kv_heads, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               seq_len, total_seq_len, head_dim, head_dim_v, num_heads, num_kv_heads, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu