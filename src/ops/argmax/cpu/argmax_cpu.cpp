#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T, typename IDX_T>
void argmax_(IDX_T *max_idx, T *max_val, const T *vals, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if (i == 0 || llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_val[0])) {
            max_val[0] = vals[i];
            max_idx[0] = static_cast<IDX_T>(i);
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<uint64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<uint64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), size);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<uint64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), size);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu