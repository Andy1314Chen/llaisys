#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t seq_len, size_t intermediate_dim) {
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < intermediate_dim; j++) {
            size_t idx = i * intermediate_dim + j;
            if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                float gate_val = llaisys::utils::cast<float>(gate[idx]);
                gate_val = gate_val / (1 + std::exp(-gate_val)); // Approximation of GELU
                float up_val = llaisys::utils::cast<float>(up[idx]);
                float swiglu_val = gate_val * up_val;
                out[idx] = llaisys::utils::cast<T>(swiglu_val);
            } else {
                // GELU for float32
                float gate_val = static_cast<float>(gate[idx]);
                gate_val = gate_val / (1 + std::exp(-gate_val)); // Approximation of GELU
                float up_val = static_cast<float>(up[idx]);
                float swiglu_val = gate_val * up_val;
                out[idx] = static_cast<T>(swiglu_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype,
            size_t seq_len, size_t intermediate_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up),
                       seq_len, intermediate_dim);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up),
                       seq_len, intermediate_dim);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up),
                       seq_len, intermediate_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu