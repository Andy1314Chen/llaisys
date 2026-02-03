#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {

    if constexpr (std::is_same_v<T, float>) {
        // Float32 优化路径
        // 策略：在 (batch, out_features) 维度并行，每个线程独立计算完整的点积

        const ptrdiff_t total_outputs = static_cast<ptrdiff_t>(batch_size * out_features);
        const ptrdiff_t in_feat = static_cast<ptrdiff_t>(in_features);
        const ptrdiff_t out_feat = static_cast<ptrdiff_t>(out_features);

#pragma omp parallel for schedule(static)
        for (ptrdiff_t idx = 0; idx < total_outputs; idx++) {
            ptrdiff_t b = idx / out_feat;
            ptrdiff_t n = idx % out_feat;

            const float *__restrict in_row = in + b * in_feat;
            const float *__restrict w_row = weight + n * in_feat;

            float sum = bias ? bias[n] : 0.0f;

            // 内层循环：编译器自动向量化（移除 omp simd 以兼容 MSVC）
            for (ptrdiff_t k = 0; k < in_feat; k++) {
                sum += in_row[k] * w_row[k];
            }

            out[b * out_feat + n] = sum;
        }
    } else {
        // FP16/BF16 路径：逐元素转换为 float 计算
        for (size_t i = 0; i < batch_size; i++) {
            const T *in_row = in + i * in_features;
            T *out_row = out + i * out_features;

            for (size_t j = 0; j < out_features; j++) {
                const T *w_row = weight + j * in_features;
                float sum = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;

                for (size_t k = 0; k < in_features; k++) {
                    sum += llaisys::utils::cast<float>(in_row[k])
                         * llaisys::utils::cast<float>(w_row[k]);
                }

                out_row[j] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype,
            size_t batch_size, size_t in_features, size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       bias ? reinterpret_cast<const float *>(bias) : nullptr,
                       batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                       batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                       batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
