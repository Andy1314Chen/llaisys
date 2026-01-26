#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Meta &meta_param, llaisysDeviceType_t dt, int did) : meta(meta_param), weights(make_weights_null()), k_cache(nullptr), v_cache(nullptr), hidden(nullptr), logits(nullptr), device_type(dt), device_ids(nullptr), ndevice(0), current_pos(0) {
    // Validate meta parameters to prevent memory allocation issues
    if (meta.nlayer == 0 || meta.hs == 0 || meta.nh == 0 || meta.nkvh == 0 || 
        meta.dh == 0 || meta.di == 0 || meta.maxseq == 0 || meta.voc == 0) {
        throw std::invalid_argument("Invalid meta parameters for Qwen2Model");
    }
    
    // Prevent excessively large allocations that could cause memory issues
    if (meta.nlayer > 1000) {
        throw std::invalid_argument("Too many layers for Qwen2Model");
    }

    // Allocate KV cache
    try {
        k_cache = new tensor_t[meta.nlayer];
        v_cache = new tensor_t[meta.nlayer];
    } catch (const std::bad_alloc& e) {
        throw;
    }

    // Initialize all tensor pointers to nullptr first to prevent undefined behavior
    for (size_t i = 0; i < meta.nlayer; ++i) {
        k_cache[i] = nullptr;
        v_cache[i] = nullptr;
    }

    for (size_t i = 0; i < meta.nlayer; ++i) {
        try {
            // K cache: [maxseq, nkvh, dh]
            k_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh},
                                        meta.dtype, device_type, did);
            if (!k_cache[i]) {
                throw std::runtime_error("Failed to create K cache tensor");
            }
            // Initialize KV cache to zero
            size_t k_cache_size = meta.maxseq * meta.nkvh * meta.dh * llaisys::utils::dsize(meta.dtype);
            std::memset(k_cache[i]->data(), 0, k_cache_size);
            
            // V cache: [maxseq, nkvh, dh]
            v_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh},
                                        meta.dtype, device_type, did);
            if (!v_cache[i]) {
                throw std::runtime_error("Failed to create V cache tensor");
            }
            // Initialize VV cache to zero
            size_t v_cache_size = meta.maxseq * meta.nkvh * meta.dh * llaisys::utils::dsize(meta.dtype);
            std::memset(v_cache[i]->data(), 0, v_cache_size);
        } catch (const std::exception& e) {
            throw;
        }
    }

    // hidden: [1, maxseq, hidden_size]
    try {
        hidden = Tensor::create({1, meta.maxseq, meta.hs}, meta.dtype, device_type, did);
        if (!hidden) {
            throw std::runtime_error("Failed to create hidden tensor");
        }
    } catch (const std::exception& e) {
        throw;
    }

    // logits: [1, vocab_size] - for single token output
    try {
        logits = Tensor::create({1, meta.voc}, meta.dtype, device_type, did);
        if (!logits) {
            throw std::runtime_error("Failed to create logits tensor");
        }
    } catch (const std::exception& e) {
        throw;
    }
}

Qwen2Model::~Qwen2Model() {
    // Clean up KV cache tensors first
    if (k_cache) {
        for (size_t i = 0; i < meta.nlayer; ++i) {
            if (k_cache[i]) {
                // Properly release the tensor
                k_cache[i] = nullptr;
            }
        }
        delete[] k_cache;
        k_cache = nullptr;
    }
    if (v_cache) {
        for (size_t i = 0; i < meta.nlayer; ++i) {
            if (v_cache[i]) {
                // Properly release the tensor
                v_cache[i] = nullptr;
            }
        }
        delete[] v_cache;
        v_cache = nullptr;
    }

    // Clean up other tensors
    if (hidden) {
        hidden = nullptr;
    }
    if (logits) {
        logits = nullptr;
    }
}

void Qwen2Model::reset_cache() {
    current_pos = 0;
    // Note: KV cache tensors remain allocated but their contents are not explicitly zeroed
    // This is efficient since they will be overwritten during forward pass anyway
}

void Qwen2Model::forward_layer(size_t layer_idx, tensor_t hidden_states, tensor_t pos_ids) {
    Qwen2Meta &meta = this->meta;
    size_t seq_len = hidden_states->shape()[1];  // [batch, seq_len, hidden_size]
    size_t hidden_size = meta.hs;
    size_t num_heads = meta.nh;
    size_t num_kv_heads = meta.nkvh;
    size_t head_dim = meta.dh;
    size_t element_size = llaisys::utils::dsize(meta.dtype);

    // Temporary tensors for intermediate results
    tensor_t normed_hidden = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);

    tensor_t q = Tensor::create(
        {seq_len, num_heads, head_dim}, meta.dtype, device_type, 0);

    tensor_t k = Tensor::create(
        {seq_len, num_kv_heads, head_dim}, meta.dtype, device_type, 0);

    tensor_t v = Tensor::create(
        {seq_len, num_kv_heads, head_dim}, meta.dtype, device_type, 0);

    tensor_t q_rope = Tensor::create(
        {seq_len, num_heads, head_dim}, meta.dtype, device_type, 0);

    tensor_t k_rope = Tensor::create(
        {seq_len, num_kv_heads, head_dim}, meta.dtype, device_type, 0);

    tensor_t attn_output = Tensor::create(
        {seq_len, num_heads, head_dim}, meta.dtype, device_type, 0);

    tensor_t gate_out = Tensor::create(
        {seq_len, meta.di}, meta.dtype, device_type, 0);

    tensor_t up_out = Tensor::create(
        {seq_len, meta.di}, meta.dtype, device_type, 0);

    tensor_t mlp_output = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);

    tensor_t residual = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);

    // ========================================
    // 1. RMSNorm (input layernorm)
    // ========================================
    // Create 2D tensor for rms_norm operation
    tensor_t hidden_states_2d = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    
    // Copy data from 3D to 2D tensor
    const std::byte* hidden_states_data = hidden_states->data();
    std::byte* hidden_states_2d_data = hidden_states_2d->data();
    
    // Get strides of the 3D source tensor
    const auto& hidden_states_strides = hidden_states->strides();
    
    // Element-wise copy with proper stride calculation
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Calculate source offset based on strides: [batch, seq_len, hidden_size]
            size_t src_offset = (0 * hidden_states_strides[0] + i * hidden_states_strides[1] + j * hidden_states_strides[2]) * element_size;
            // Calculate destination offset: [seq_len, hidden_size]
            size_t dst_offset = (i * hidden_size + j) * element_size;
            std::memcpy(hidden_states_2d_data + dst_offset, hidden_states_data + src_offset, element_size);
        }
    }
    
    // Create 2D tensor for normed_hidden operation
    tensor_t normed_hidden_2d = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    
    // Apply RMSNorm
    ops::rms_norm(normed_hidden_2d, hidden_states_2d,
                  weights.attn_norm_w[layer_idx], meta.epsilon);
    
    // Copy data from 2D to 3D tensor
    const std::byte* normed_hidden_2d_data = normed_hidden_2d->data();
    std::byte* normed_hidden_data = normed_hidden->data();
    
    // Get strides of the 3D destination tensor
    const auto& normed_hidden_strides = normed_hidden->strides();
    
    // Element-wise copy with proper stride calculation
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Calculate source offset: [seq_len, hidden_size]
            size_t src_offset = (i * hidden_size + j) * element_size;
            // Calculate destination offset based on strides: [batch, seq_len, hidden_size]
            size_t dst_offset = (0 * normed_hidden_strides[0] + i * normed_hidden_strides[1] + j * normed_hidden_strides[2]) * element_size;
            std::memcpy(normed_hidden_data + dst_offset, normed_hidden_2d_data + src_offset, element_size);
        }
    }

    // ========================================
    // 2. QKV Projection
    // ========================================
    // Create 2D tensors for linear projections (expected by ops::linear)
    // Input shape: [seq_len, hidden_size]
    // Output shape: [seq_len, num_heads*head_dim]
    tensor_t normed_hidden_2d_proj = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    
    // Copy data from 3D to 2D tensor
    const std::byte* src_data = normed_hidden->data();
    std::byte* dst_data = normed_hidden_2d_proj->data();
    
    // Get strides of the 3D source tensor
    const auto& src_strides = normed_hidden->strides();
    
    // Element-wise copy with proper stride calculation
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Calculate source offset based on strides: [batch, seq_len, hidden_size]
            size_t src_offset = (0 * src_strides[0] + i * src_strides[1] + j * src_strides[2]) * element_size;
            // Calculate destination offset: [seq_len, hidden_size]
            size_t dst_offset = (i * hidden_size + j) * element_size;
            std::memcpy(dst_data + dst_offset, src_data + src_offset, element_size);
        }
    }
    
    // Debug: verify the copy was successful
    // Removed debug print
    
    // Create 2D output tensors for linear projections
    tensor_t q_flat_2d = Tensor::create(
        {seq_len, num_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t k_flat_2d = Tensor::create(
        {seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t v_flat_2d = Tensor::create(
        {seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);

    // Perform linear projections
    ops::linear(q_flat_2d, normed_hidden_2d_proj,
                weights.attn_q_w[layer_idx], weights.attn_q_b[layer_idx]);
    ops::linear(k_flat_2d, normed_hidden_2d_proj,
                weights.attn_k_w[layer_idx], weights.attn_k_b[layer_idx]);
    ops::linear(v_flat_2d, normed_hidden_2d_proj,
                weights.attn_v_w[layer_idx], weights.attn_v_b[layer_idx]);

    // Create 3D tensors for subsequent operations
    tensor_t q_flat = Tensor::create(
        {1, seq_len, num_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t k_flat = Tensor::create(
        {1, seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t v_flat = Tensor::create(
        {1, seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);

    // Copy data from 2D to 3D tensors
    // Get strides of the 3D destination tensors
    const auto& q_flat_strides = q_flat->strides();
    const auto& k_flat_strides = k_flat->strides();
    const auto& v_flat_strides = v_flat->strides();
    
    // Copy q_flat_2d to q_flat
    src_data = q_flat_2d->data();
    dst_data = q_flat->data();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < num_heads * head_dim; ++j) {
            // Calculate source offset: [seq_len, num_heads*head_dim]
            size_t src_offset = (i * num_heads * head_dim + j) * element_size;
            // Calculate destination offset based on strides: [batch, seq_len, num_heads*head_dim]
            size_t dst_offset = (0 * q_flat_strides[0] + i * q_flat_strides[1] + j * q_flat_strides[2]) * element_size;
            std::memcpy(dst_data + dst_offset, src_data + src_offset, element_size);
        }
    }
    
    // Copy k_flat_2d to k_flat
    src_data = k_flat_2d->data();
    dst_data = k_flat->data();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < num_kv_heads * head_dim; ++j) {
            // Calculate source offset: [seq_len, num_kv_heads*head_dim]
            size_t src_offset = (i * num_kv_heads * head_dim + j) * element_size;
            // Calculate destination offset based on strides: [batch, seq_len, num_kv_heads*head_dim]
            size_t dst_offset = (0 * k_flat_strides[0] + i * k_flat_strides[1] + j * k_flat_strides[2]) * element_size;
            std::memcpy(dst_data + dst_offset, src_data + src_offset, element_size);
        }
    }
    
    // Copy v_flat_2d to v_flat
    src_data = v_flat_2d->data();
    dst_data = v_flat->data();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < num_kv_heads * head_dim; ++j) {
            // Calculate source offset: [seq_len, num_kv_heads*head_dim]
            size_t src_offset = (i * num_kv_heads * head_dim + j) * element_size;
            // Calculate destination offset based on strides: [batch, seq_len, num_kv_heads*head_dim]
            size_t dst_offset = (0 * v_flat_strides[0] + i * v_flat_strides[1] + j * v_flat_strides[2]) * element_size;
            std::memcpy(dst_data + dst_offset, src_data + src_offset, element_size);
        }
    }

    // Copy flat results to reshaped tensors with proper data layout conversion
    // q_flat: [1, seq_len, num_heads*head_dim] -> q: [seq_len, num_heads, head_dim]
    // k_flat: [1, seq_len, num_kv_heads*head_dim] -> k: [seq_len, num_kv_heads, head_dim]
    // v_flat: [1, seq_len, num_kv_heads*head_dim] -> v: [seq_len, num_kv_heads, head_dim]
    
    // Copy q_flat to q with proper layout
    const std::byte* q_flat_data = q_flat->data();
    std::byte* q_data = q->data();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < num_heads; ++j) {
            for (size_t k = 0; k < head_dim; ++k) {
                // Calculate source offset based on strides: [batch, seq_len, num_heads*head_dim]
                size_t src_offset = (0 * q_flat_strides[0] + i * q_flat_strides[1] + (j * head_dim + k) * q_flat_strides[2]) * element_size;
                // Calculate destination offset: [seq_len, num_heads, head_dim]
                size_t dst_offset = (i * num_heads * head_dim + j * head_dim + k) * element_size;
                std::memcpy(q_data + dst_offset, q_flat_data + src_offset, element_size);
            }
        }
    }
    
    // Copy k_flat to k with proper layout
    const std::byte* k_flat_data = k_flat->data();
    std::byte* k_data = k->data();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < num_kv_heads; ++j) {
            for (size_t k = 0; k < head_dim; ++k) {
                // Calculate source offset based on strides: [batch, seq_len, num_kv_heads*head_dim]
                size_t src_offset = (0 * k_flat_strides[0] + i * k_flat_strides[1] + (j * head_dim + k) * k_flat_strides[2]) * element_size;
                // Calculate destination offset: [seq_len, num_kv_heads, head_dim]
                size_t dst_offset = (i * num_kv_heads * head_dim + j * head_dim + k) * element_size;
                std::memcpy(k_data + dst_offset, k_flat_data + src_offset, element_size);
            }
        }
    }
    
    // Copy v_flat to v with proper layout
    const std::byte* v_flat_data = v_flat->data();
    std::byte* v_data = v->data();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < num_kv_heads; ++j) {
            for (size_t k = 0; k < head_dim; ++k) {
                // Calculate source offset based on strides: [batch, seq_len, num_kv_heads*head_dim]
                size_t src_offset = (0 * v_flat_strides[0] + i * v_flat_strides[1] + (j * head_dim + k) * v_flat_strides[2]) * element_size;
                // Calculate destination offset: [seq_len, num_kv_heads, head_dim]
                size_t dst_offset = (i * num_kv_heads * head_dim + j * head_dim + k) * element_size;
                std::memcpy(v_data + dst_offset, v_flat_data + src_offset, element_size);
            }
        }
    }

    // ========================================
    // 3. RoPE (Rotary Position Embedding)
    // ========================================
    // Create 1D pos_ids tensor for rope operation
    tensor_t pos_ids_1d = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type, 0);
    
    // Copy data from 2D pos_ids to 1D pos_ids
    const std::byte* pos_ids_data = pos_ids->data();
    std::byte* pos_ids_1d_data = pos_ids_1d->data();
    size_t pos_element_size = llaisys::utils::dsize(LLAISYS_DTYPE_I64);
    
    // Get strides of the 2D pos_ids tensor
    const auto& pos_ids_strides = pos_ids->strides();
    
    // Element-wise copy with proper stride calculation
    for (size_t i = 0; i < seq_len; ++i) {
        // Calculate source offset based on strides: [batch, seq_len]
        size_t src_offset = (0 * pos_ids_strides[0] + i * pos_ids_strides[1]) * pos_element_size;
        // Calculate destination offset: [seq_len]
        size_t dst_offset = i * pos_element_size;
        std::memcpy(pos_ids_1d_data + dst_offset, pos_ids_data + src_offset, pos_element_size);
    }
    
    // Apply RoPE to K (use the 1D pos_ids tensor)
    ops::rope(k_rope, k, pos_ids_1d, meta.theta);
    
    // Apply RoPE to Q (use the 1D pos_ids tensor)
    ops::rope(q_rope, q, pos_ids_1d, meta.theta);

    // ========================================
    // 3. KV Cache Update & Attention
    // ========================================
    
    // // DEBUG
    // fprintf(stderr, "DEBUG forward_layer: layer=%zu, seq_len=%zu, current_pos=%zu\n",
    //         layer_idx, seq_len, this->current_pos);
    
    // Get the KV cache tensors for the current layer
    tensor_t layer_k_cache = this->k_cache[layer_idx];
    tensor_t layer_v_cache = this->v_cache[layer_idx];
    
    // Calculate the size of data to copy (seq_len * nkvh * dh)
    size_t kv_elements = seq_len * num_kv_heads * head_dim;
    size_t kv_bytes = kv_elements * element_size;

    // 1. Write current k_rope and v into cache
    // We need to write into the cache at offset: current_pos
    // k_cache shape is [maxseq, nkvh, dh]
    // We take a slice of the cache corresponding to the current time steps
    tensor_t k_cache_dest = layer_k_cache->slice(0, this->current_pos, this->current_pos + seq_len);
    tensor_t v_cache_dest = layer_v_cache->slice(0, this->current_pos, this->current_pos + seq_len);

    // Copy data from k_rope and v to the cache slice
    // Note: K needs RoPE, but V does not need RoPE
    // Using memcpy_sync for Device-to-Device copy
    auto runtime_api = llaisys::core::context().runtime().api();

    runtime_api->memcpy_sync(
        k_cache_dest->data(), k_rope->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

    runtime_api->memcpy_sync(
        v_cache_dest->data(), v->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

    // 2. Prepare inputs for Self-Attention
    // We need to attend to all tokens up to current_pos + seq_len
    size_t total_seq_len = this->current_pos + seq_len;
    
    // // DEBUG
    // fprintf(stderr, "DEBUG KV cache: current_pos=%zu, seq_len=%zu, total_seq_len=%zu\n",
    //         this->current_pos, seq_len, total_seq_len);
    
    // Create views of the cache containing all valid history + current
    tensor_t k_history = layer_k_cache->slice(0, 0, total_seq_len);
    tensor_t v_history = layer_v_cache->slice(0, 0, total_seq_len);

    // Perform Scaled Dot-Product Attention
    // Output: [seq_len, num_heads, head_dim]
    // Note: ops::self_attention implementation should handle GQA (grouping repeats) if needed
    // Q is [seq_len, num_heads, head_dim]
    // K, V are [total_seq_len, num_kv_heads, head_dim]
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // IMPORTANT: Ensure self_attention handles the causal mask correctly inside,
    // or pass a mask if the API requires it.
    ops::self_attention(attn_output, q_rope, k_history, v_history, scale);

    // ========================================
    // 4. Output Projection
    // ========================================
    // Flatten attn_output for linear projection
    // [seq_len, num_heads, head_dim] -> [seq_len, hidden_size]
    tensor_t attn_output_flat = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    size_t attn_size = seq_len * num_heads * head_dim * llaisys::utils::dsize(meta.dtype);
    llaisys::core::context().runtime().api()->memcpy_sync(
        attn_output_flat->data(), attn_output->data(), attn_size, LLAISYS_MEMCPY_H2H);

    // Create 2D output tensor for linear projection (linear expects 2D)
    tensor_t attn_out_proj = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    ops::linear(attn_out_proj, attn_output_flat,
                weights.attn_o_w[layer_idx], nullptr);

    // 7. Residual Connection (Attention)
    // ========================================
    // Convert attn_out_proj from 2D to 3D for residual connection
    tensor_t attn_out_proj_3d = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);

    // Copy data from 2D to 3D tensor
    const std::byte* attn_out_proj_data = attn_out_proj->data();
    std::byte* attn_out_proj_3d_data = attn_out_proj_3d->data();

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Source: [seq_len, hidden_size]
            size_t src_offset = (i * hidden_size + j) * element_size;
            // Destination: [batch, seq_len, hidden_size] - use hidden_states strides
            const auto& hidden_strides = hidden_states->strides();
            size_t dst_offset = (0 * hidden_strides[0] + i * hidden_strides[1] + j * hidden_strides[2]) * element_size;
            std::memcpy(attn_out_proj_3d_data + dst_offset, attn_out_proj_data + src_offset, element_size);
        }
    }

    ops::add(hidden_states, hidden_states, attn_out_proj_3d);

    // ========================================
    // 8. RMSNorm (post attention layernorm)
    // ========================================
    // Create 2D tensor for rms_norm operation
    tensor_t hidden_states_2d_mlp = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    
    // Copy data from 3D to 2D tensor
    const std::byte* hidden_states_data_mlp = hidden_states->data();
    std::byte* hidden_states_2d_mlp_data = hidden_states_2d_mlp->data();
    
    // Get strides of the 3D source tensor
    const auto& hidden_states_strides_mlp = hidden_states->strides();
    
    // Element-wise copy with proper stride calculation
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Calculate source offset based on strides: [batch, seq_len, hidden_size]
            size_t src_offset = (0 * hidden_states_strides_mlp[0] + i * hidden_states_strides_mlp[1] + j * hidden_states_strides_mlp[2]) * element_size;
            // Calculate destination offset: [seq_len, hidden_size]
            size_t dst_offset = (i * hidden_size + j) * element_size;
            std::memcpy(hidden_states_2d_mlp_data + dst_offset, hidden_states_data_mlp + src_offset, element_size);
        }
    }
    
    // Create 2D output tensor for RMSNorm
    tensor_t normed_hidden_2d_mlp = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    
    // Apply RMSNorm
    ops::rms_norm(normed_hidden_2d_mlp, hidden_states_2d_mlp,
                  weights.mlp_norm_w[layer_idx], meta.epsilon);
    
    // Copy data from 2D to 3D tensor
    const std::byte* normed_hidden_2d_mlp_data = normed_hidden_2d_mlp->data();
    std::byte* normed_hidden_data_mlp = normed_hidden->data();
    
    // Get strides of the 3D destination tensor
    const auto& normed_hidden_strides_mlp = normed_hidden->strides();
    
    // Element-wise copy with proper stride calculation
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Calculate source offset: [seq_len, hidden_size]
            size_t src_offset = (i * hidden_size + j) * element_size;
            // Calculate destination offset based on strides: [batch, seq_len, hidden_size]
            size_t dst_offset = (0 * normed_hidden_strides_mlp[0] + i * normed_hidden_strides_mlp[1] + j * normed_hidden_strides_mlp[2]) * element_size;
            std::memcpy(normed_hidden_data_mlp + dst_offset, normed_hidden_2d_mlp_data + src_offset, element_size);
        }
    }

    // ========================================
    // 9. SwiGLU MLP
    // ========================================
    // Use 2D normed_hidden_2d_mlp for linear operations (linear expects 2D input)
    ops::linear(gate_out, normed_hidden_2d_mlp,
                weights.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_out, normed_hidden_2d_mlp,
                weights.mlp_up_w[layer_idx], nullptr);

    // Compute SwiGLU activation
    ops::swiglu(gate_out, gate_out, up_out);

    // Apply down projection
    ops::linear(mlp_output, gate_out,
                weights.mlp_down_w[layer_idx], nullptr);

    // 10. Residual Connection (MLP)
    // ========================================
    // Convert mlp_output from 2D to 3D for residual connection
    tensor_t mlp_output_3d = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);

    // Copy data from 2D to 3D tensor
    const std::byte* mlp_output_data = mlp_output->data();
    std::byte* mlp_output_3d_data = mlp_output_3d->data();

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            // Source: [seq_len, hidden_size]
            size_t src_offset = (i * hidden_size + j) * element_size;
            // Destination: [batch, seq_len, hidden_size] - use hidden_states strides
            const auto& hidden_strides = hidden_states->strides();
            size_t dst_offset = (0 * hidden_strides[0] + i * hidden_strides[1] + j * hidden_strides[2]) * element_size;
            std::memcpy(mlp_output_3d_data + dst_offset, mlp_output_data + src_offset, element_size);
        }
    }

    ops::add(hidden_states, hidden_states, mlp_output_3d);

    // Note: current_pos is updated by the caller (Qwen2ModelInfer) after all layers are processed
}


Qwen2Model *Qwen2ModelCreate(const Qwen2Meta *meta, llaisysDeviceType_t device_type, int *device_ids, int ndevice) {
    Qwen2Model *model = new Qwen2Model(*meta, device_type, device_ids ? device_ids[0] : 0);
    return model;
}

void Qwen2ModelDestroy(Qwen2Model *model) {
    if (model) {
        // Free KV cache
        if (model->k_cache) {
            delete[] model->k_cache;
            model->k_cache = nullptr;
        }
        if (model->v_cache) {
            delete[] model->v_cache;
            model->v_cache = nullptr;
        }
        delete model;
    }
}

struct Qwen2Weights *Qwen2ModelWeights(Qwen2Model *model) {
    return &model->weights;
}

int64_t Qwen2ModelInfer(Qwen2Model *model, int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature) {
    // Detect prefill vs decode mode
    bool is_prefill = (model->current_pos == 0);

    // For decode mode, only process the last token
    size_t actual_ntoken = is_prefill ? ntoken : 1;
    int64_t *actual_token_ids = is_prefill ? token_ids : token_ids;  // FIX: Don't offset, use token_ids directly

    // DEBUG
    // fprintf(stderr, "DEBUG: is_prefill=%s, current_pos=%zu, ntoken=%zu, actual_ntoken=%zu\n",
    //         is_prefill ? "true" : "false", model->current_pos, ntoken, actual_ntoken);
    // fprintf(stderr, "DEBUG: actual_token_ids[0]=%ld\n", actual_token_ids[0]);
    
    // 1. Embed input tokens
    // token_indices should be 1D [actual_ntoken] for embedding operation
    tensor_t token_indices = Tensor::create(
        {actual_ntoken}, LLAISYS_DTYPE_I64, model->device_type, 0);
    token_indices->load(actual_token_ids);
    // token_indices->debug();

    // Create 2D hidden_states tensor for embedding (expected by ops::embedding)
    tensor_t hidden_states_2d = Tensor::create(
        {actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);

    // Embedding lookup: hidden_states_2d[token] = in_embed[token]
    ops::embedding(hidden_states_2d, token_indices, model->weights.in_embed);
    // hidden_states_2d->debug();

    // Reshape to 3D [1, actual_ntoken, model->meta.hs] for subsequent layers
    tensor_t hidden_states = Tensor::create(
        {1, actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);

    // Copy data from 2D to 3D tensor
    size_t element_size = llaisys::utils::dsize(model->meta.dtype);
    const std::byte* src_data = hidden_states_2d->data();
    std::byte* dst_data = hidden_states->data();
    size_t total_elements = actual_ntoken * model->meta.hs;
    for (size_t i = 0; i < total_elements; ++i) {
        std::memcpy(dst_data + i * element_size, src_data + i * element_size, element_size);
    }

    // 2. Create position IDs [current_pos, current_pos+1, ..., current_pos+actual_ntoken-1]
    // Note: pos_ids should be 2D [1, actual_ntoken] for rope operation
    tensor_t pos_ids = Tensor::create({1, actual_ntoken}, LLAISYS_DTYPE_I64, model->device_type, 0);
    std::vector<int64_t> pos_data(actual_ntoken);
    for (size_t i = 0; i < actual_ntoken; ++i) {
        pos_data[i] = static_cast<int64_t>(model->current_pos + i);
    }
    pos_ids->load(pos_data.data());
    
    // // DEBUG
    // fprintf(stderr, "DEBUG: pos_ids=[%ld]\n", pos_data[0]);
    // pos_ids->debug();

    // 3. Forward pass through all layers
    for (size_t layer_idx = 0; layer_idx < model->meta.nlayer; ++layer_idx) {
        model->forward_layer(layer_idx, hidden_states, pos_ids);
    }

    // 4. Final layer norm
    // IMPORTANT: rms_norm expects 2D tensor [batch_size, feature_size]
    // We need to reshape hidden_states from [1, actual_ntoken, model->meta.hs] to [actual_ntoken, model->meta.hs]
    tensor_t final_hidden_states_2d = Tensor::create(
        {actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    tensor_t normed_output_2d = Tensor::create(
        {actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    
    // Copy data from 3D to 2D tensor
    const std::byte* src_data_final = hidden_states->data();
    std::byte* dst_data_final = final_hidden_states_2d->data();
    const auto& hidden_strides_final = hidden_states->strides();
    
    for (size_t i = 0; i < actual_ntoken; ++i) {
        for (size_t j = 0; j < model->meta.hs; ++j) {
            size_t src_offset = (0 * hidden_strides_final[0] + i * hidden_strides_final[1] + j * hidden_strides_final[2]) * element_size;
            size_t dst_offset = (i * model->meta.hs + j) * element_size;
            std::memcpy(dst_data_final + dst_offset, src_data_final + src_offset, element_size);
        }
    }
    
    ops::rms_norm(normed_output_2d, final_hidden_states_2d,
                  model->weights.out_norm_w, model->meta.epsilon);
    
    // Create 3D normed_output tensor
    tensor_t normed_output = Tensor::create(
        {1, actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    
    // Copy data from 2D back to 3D tensor
    const std::byte* normed_2d_data = normed_output_2d->data();
    std::byte* normed_3d_data = normed_output->data();
    const auto& normed_strides = normed_output->strides();
    
    for (size_t i = 0; i < actual_ntoken; ++i) {
        for (size_t j = 0; j < model->meta.hs; ++j) {
            size_t src_offset = (i * model->meta.hs + j) * element_size;
            size_t dst_offset = (0 * normed_strides[0] + i * normed_strides[1] + j * normed_strides[2]) * element_size;
            std::memcpy(normed_3d_data + dst_offset, normed_2d_data + src_offset, element_size);
        }
    }

    // 5. Project to vocab logits (take last token's output)
    // Use the correct slice method: slice(dim, start, end)
    tensor_t last_hidden_slice = normed_output->slice(1, actual_ntoken-1, actual_ntoken); // Get last token along sequence dimension

    // DEBUG: print logits before argmax
    // fprintf(stderr, "DEBUG: actual_ntoken=%zu, slice from %zu to %zu\n", actual_ntoken, actual_ntoken-1, actual_ntoken);
    
    // The slice returns [1, 1, 1536], but Linear expects [1, 1536]
    // We need to squeeze the sequence dimension
    tensor_t last_hidden = Tensor::create(
        {1, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    
    // FIX: Use runtime API for memory copy instead of std::memcpy loop
    // This handles Device-to-Device copy correctly and respects the slice offset
    size_t hidden_bytes = model->meta.hs * llaisys::utils::dsize(model->meta.dtype);
    
    // Copy contiguous block of memory
    // last_hidden_slice->data() points to the start of the sliced region (thanks to correct slice implementation)
    llaisys::core::context().runtime().api()->memcpy_sync(
        last_hidden->data(),       // dst: new contiguous tensor
        last_hidden_slice->data(), // src: sliced tensor view
        hidden_bytes,              // size: one vector of hidden_size
        LLAISYS_MEMCPY_D2D         // type: Device to Device
    );
    
    tensor_t logits = Tensor::create(
        {1, model->meta.voc}, model->meta.dtype, model->device_type, 0);
    ops::linear(logits, last_hidden, model->weights.out_embed, nullptr);

    // 6. Get next token using sampling or argmax
    int64_t next_token;
    
    // Create a CPU tensor to hold the result index
    tensor_t max_idx_cpu = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    size_t idx_size = llaisys::utils::dsize(LLAISYS_DTYPE_I64);

    if (top_k == 1) {
        // Greedy decoding (argmax)
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, model->device_type, 0);
        tensor_t max_val = Tensor::create({1}, model->meta.dtype, model->device_type, 0);
        ops::argmax(max_idx, max_val, logits);
        
        // Fix: Copy result from device to host before reading
        llaisys::core::context().runtime().api()->memcpy_sync(
            max_idx_cpu->data(), max_idx->data(), idx_size, LLAISYS_MEMCPY_D2H);
            
        const std::byte *data_ptr = max_idx_cpu->data();
        next_token = *reinterpret_cast<const int64_t *>(data_ptr);
    } else {
        // Implement sampling with top_k and top_p
        // For now, just use argmax as a placeholder
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, model->device_type, 0);
        tensor_t max_val = Tensor::create({1}, model->meta.dtype, model->device_type, 0);
        ops::argmax(max_idx, max_val, logits);
        
        // Fix: Copy result from device to host before reading
        llaisys::core::context().runtime().api()->memcpy_sync(
            max_idx_cpu->data(), max_idx->data(), idx_size, LLAISYS_MEMCPY_D2H);

        const std::byte *data_ptr = max_idx_cpu->data();
        next_token = *reinterpret_cast<const int64_t *>(data_ptr);
    }

    // 8. Update position counter
    model->current_pos += actual_ntoken;

    return next_token;
}

void Qwen2ModelResetCache(Qwen2Model *model) {
    if (model) {
        model->reset_cache();
    }
}

} // namespace llaisys::models
