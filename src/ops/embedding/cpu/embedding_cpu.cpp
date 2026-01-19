#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight,
                const std::vector<size_t> &out_shape,
                const std::vector<size_t> &index_shape,
                const std::vector<size_t> &weight_shape) {
    size_t batch_size = index_shape[0];
    size_t embed_dim = weight_shape[1];
    for (size_t i = 0; i < batch_size; i++) {
        int64_t idx = index[i];
        for (size_t j = 0; j < embed_dim; j++) {
            out[i * embed_dim + j] = weight[idx * embed_dim + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype,
               const std::vector<size_t> &out_shape,
               const std::vector<size_t> &index_shape,
               const std::vector<size_t> &weight_shape) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight),
                          out_shape, index_shape, weight_shape);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          out_shape, index_shape, weight_shape);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          out_shape, index_shape, weight_shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu