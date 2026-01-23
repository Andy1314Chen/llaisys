#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);    
    if (attn_val->shape()[1] == 1) {  // Debug: print info for single-token case
        fprintf(stderr, "[self_attention] seq_len=%zu, total_seq_len=%zu, head_dim=%zu, num_heads=%zu\n",
                attn_val->shape()[1], k->shape()[0], q->shape()[2], q->shape()[2]/q->shape()[1]/v->shape()[1]);
        fprintf(stderr, "[self_attention] k shape=[%zu,%zu,%zu], v shape=[%zu,%zu,%zu]\n",
                k->shape()[0], k->shape()[1], k->shape()[2], v->shape()[0], v->shape()[1], v->shape()[2]);
        fflush(stderr);
    }
        ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-Attention: all tensors must be contiguous.");
    size_t seq_len = q->shape()[0];
    size_t total_seq_len = k->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t head_dim_v = v->shape()[2];
    size_t num_kv_heads = k->shape()[1];

    // always support cpu calculation
    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seq_len, total_seq_len, head_dim, head_dim_v,
                                   num_heads, num_kv_heads, scale);
    }

    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());
    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seq_len, total_seq_len, head_dim, head_dim_v,
                                   num_heads, num_kv_heads, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
