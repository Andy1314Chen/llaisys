#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    // Manual loop unrolling and optimizations for better performance
    // Works on all CPU architectures without requiring SIMD intrinsics

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // FP16/BF16: convert to float, compute, convert back
        for (size_t i = 0; i < batch_size; i++) {
            const T *in_row = in + i * in_features;
            T *out_row = out + i * out_features;

            for (size_t j = 0; j < out_features; j++) {
                const T *w_row = weight + j * in_features;
                float sum = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;

                // Manual loop unrolling for better instruction-level parallelism
                size_t k = 0;
                const float *in_row_f = reinterpret_cast<const float*>(in_row);
                const float *w_row_f = reinterpret_cast<const float*>(w_row);

                // Unroll by 8 for better performance
                for (; k + 8 <= in_features; k += 8) {
                    sum += in_row_f[k] * w_row_f[k] +
                           in_row_f[k+1] * w_row_f[k+1] +
                           in_row_f[k+2] * w_row_f[k+2] +
                           in_row_f[k+3] * w_row_f[k+3] +
                           in_row_f[k+4] * w_row_f[k+4] +
                           in_row_f[k+5] * w_row_f[k+5] +
                           in_row_f[k+6] * w_row_f[k+6] +
                           in_row_f[k+7] * w_row_f[k+7];
                }

                // Handle remaining elements
                for (; k < in_features; k++) {
                    sum += in_row_f[k] * w_row_f[k];
                }

                out_row[j] = llaisys::utils::cast<T>(sum);
            }
        }
    } else {
        // Float32: direct computation with manual unrolling
        for (size_t i = 0; i < batch_size; i++) {
            const T *in_row = in + i * in_features;
            T *out_row = out + i * out_features;

            for (size_t j = 0; j < out_features; j++) {
                const T *w_row = weight + j * in_features;
                T sum = bias ? bias[j] : static_cast<T>(0);

                size_t k = 0;

                // Unroll by 8 for better instruction-level parallelism
                for (; k + 8 <= in_features; k += 8) {
                    sum += in_row[k] * w_row[k] +
                           in_row[k+1] * w_row[k+1] +
                           in_row[k+2] * w_row[k+2] +
                           in_row[k+3] * w_row[k+3] +
                           in_row[k+4] * w_row[k+4] +
                           in_row[k+5] * w_row[k+5] +
                           in_row[k+6] * w_row[k+6] +
                           in_row[k+7] * w_row[k+7];
                }

                // Handle remaining elements
                for (; k < in_features; k++) {
                    sum += in_row[k] * w_row[k];
                }

                out_row[j] = sum;
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