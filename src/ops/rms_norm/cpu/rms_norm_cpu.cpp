#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t batch_size, size_t feature_size, float eps) {
    for (size_t i = 0; i < batch_size; i++) {
        // compute rms
        float sum_sq = 0.0f;
        for (size_t j = 0; j < feature_size; j++) {
            float val = llaisys::utils::cast<float>(in[i * feature_size + j]);
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / feature_size) + eps; // add eps to avoid division by zero
        for (size_t j = 0; j < feature_size; j++) {
            float val = llaisys::utils::cast<float>(in[i * feature_size + j]);
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * feature_size + j] = llaisys::utils::cast<T>(val * llaisys::utils::cast<float>(weight[j]) / rms);
            } else {
                out[i * feature_size + j] = llaisys::utils::cast<T>(val * weight[j] / rms);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype,
              size_t batch_size, size_t feature_size, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         batch_size, feature_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         batch_size, feature_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         batch_size, feature_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu