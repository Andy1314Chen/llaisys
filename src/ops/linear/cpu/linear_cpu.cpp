#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < out_features; j++) {

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // for bf16 and f16, use float for accumulation to improve precision
                float sum = bias ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
                for (size_t k = 0; k < in_features; k++) {
                    sum += llaisys::utils::cast<float>(in[i * in_features + k]) * llaisys::utils::cast<float>(weight[j * in_features + k]);
                }
                out[i * out_features + j] = llaisys::utils::cast<T>(sum); // TODO: activation
                continue;
            } else {
                T sum = bias ? bias[j] : static_cast<T>(0);
                for (size_t k = 0; k < in_features; k++) {
                    sum += in[i * in_features + k] * weight[j * in_features + k];
                }
                out[i * out_features + j] = sum; // TODO: activation
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