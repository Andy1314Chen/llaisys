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
    } catch (const std::bad_alloc&) {
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
        } catch (const std::exception&) {
            throw;
        }
    }

    // hidden: [1, maxseq, hidden_size]
    try {
        hidden = Tensor::create({1, meta.maxseq, meta.hs}, meta.dtype, device_type, did);
        if (!hidden) {
            throw std::runtime_error("Failed to create hidden tensor");
        }
    } catch (const std::exception&) {
        throw;
    }

    // logits: [1, vocab_size] - for single token output
    try {
        logits = Tensor::create({1, meta.voc}, meta.dtype, device_type, did);
        if (!logits) {
            throw std::runtime_error("Failed to create logits tensor");
        }
    } catch (const std::exception&) {
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

    // For batch=1 tensors, we can use view() to avoid data copies
    // The data is contiguous in memory, so reshaping doesn't require copying

    // ========================================
    // 1. RMSNorm (input layernorm)
    // ========================================
    // View hidden_states as 2D: [1, seq_len, hidden_size] -> [seq_len, hidden_size]
    // This is a zero-copy operation since batch=1
    tensor_t hidden_states_2d = hidden_states->view({seq_len, hidden_size});
    tensor_t normed_hidden_2d = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);

    // Apply RMSNorm
    ops::rms_norm(normed_hidden_2d, hidden_states_2d,
                  weights.attn_norm_w[layer_idx], meta.epsilon);

    // ========================================
    // 2. QKV Projection
    // ========================================
    // Create 2D output tensors for linear projections
    tensor_t q_flat_2d = Tensor::create(
        {seq_len, num_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t k_flat_2d = Tensor::create(
        {seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t v_flat_2d = Tensor::create(
        {seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);

    // Perform linear projections
    ops::linear(q_flat_2d, normed_hidden_2d,
                weights.attn_q_w[layer_idx], weights.attn_q_b[layer_idx]);
    ops::linear(k_flat_2d, normed_hidden_2d,
                weights.attn_k_w[layer_idx], weights.attn_k_b[layer_idx]);
    ops::linear(v_flat_2d, normed_hidden_2d,
                weights.attn_v_w[layer_idx], weights.attn_v_b[layer_idx]);

    // Reshape flat tensors to [seq_len, num_heads, head_dim] using view (zero-copy)
    tensor_t q = q_flat_2d->view({seq_len, num_heads, head_dim});
    tensor_t k = k_flat_2d->view({seq_len, num_kv_heads, head_dim});
    tensor_t v = v_flat_2d->view({seq_len, num_kv_heads, head_dim});

    // ========================================
    // 3. RoPE (Rotary Position Embedding)
    // ========================================
    // View pos_ids as 1D: [1, seq_len] -> [seq_len] (zero-copy for batch=1)
    tensor_t pos_ids_1d = pos_ids->view({seq_len});

    // Create tensors for RoPE outputs
    tensor_t q_rope = Tensor::create(
        {seq_len, num_heads, head_dim}, meta.dtype, device_type, 0);
    tensor_t k_rope = Tensor::create(
        {seq_len, num_kv_heads, head_dim}, meta.dtype, device_type, 0);
    tensor_t attn_output = Tensor::create(
        {seq_len, num_heads, head_dim}, meta.dtype, device_type, 0);

    // Apply RoPE to K and Q
    ops::rope(k_rope, k, pos_ids_1d, meta.theta);
    ops::rope(q_rope, q, pos_ids_1d, meta.theta);

    // ========================================
    // 4. KV Cache Update & Attention
    // ========================================
    // Get the KV cache tensors for the current layer
    tensor_t layer_k_cache = this->k_cache[layer_idx];
    tensor_t layer_v_cache = this->v_cache[layer_idx];

    size_t kv_elements = seq_len * num_kv_heads * head_dim;
    size_t kv_bytes = kv_elements * llaisys::utils::dsize(meta.dtype);

    // Write current k_rope and v into cache
    tensor_t k_cache_dest = layer_k_cache->slice(0, this->current_pos, this->current_pos + seq_len);
    tensor_t v_cache_dest = layer_v_cache->slice(0, this->current_pos, this->current_pos + seq_len);

    auto runtime_api = llaisys::core::context().runtime().api();
    runtime_api->memcpy_sync(
        k_cache_dest->data(), k_rope->data(), kv_bytes, LLAISYS_MEMCPY_D2D);
    runtime_api->memcpy_sync(
        v_cache_dest->data(), v->data(), kv_bytes, LLAISYS_MEMCPY_D2D);

    // Prepare inputs for Self-Attention
    size_t total_seq_len = this->current_pos + seq_len;
    tensor_t k_history = layer_k_cache->slice(0, 0, total_seq_len);
    tensor_t v_history = layer_v_cache->slice(0, 0, total_seq_len);

    // Perform Scaled Dot-Product Attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ops::self_attention(attn_output, q_rope, k_history, v_history, scale);

    // ========================================
    // 5. Output Projection
    // ========================================
    // View attn_output as 2D: [seq_len, num_heads, head_dim] -> [seq_len, hidden_size] (zero-copy)
    tensor_t attn_output_flat = attn_output->view({seq_len, hidden_size});

    tensor_t attn_out_proj = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    ops::linear(attn_out_proj, attn_output_flat,
                weights.attn_o_w[layer_idx], nullptr);

    // ========================================
    // 6. Residual Connection (Attention)
    // ========================================
    // Create 3D residual tensor using memcpy for efficient bulk copy
    tensor_t attn_out_proj_3d = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);
    size_t residual_bytes = seq_len * hidden_size * llaisys::utils::dsize(meta.dtype);
    runtime_api->memcpy_sync(
        attn_out_proj_3d->data(), attn_out_proj->data(), residual_bytes, LLAISYS_MEMCPY_H2H);
    ops::add(hidden_states, hidden_states, attn_out_proj_3d);

    // ========================================
    // 7. RMSNorm (post attention layernorm)
    // ========================================
    // View hidden_states as 2D: [1, seq_len, hidden_size] -> [seq_len, hidden_size] (zero-copy)
    tensor_t hidden_states_2d_mlp = hidden_states->view({seq_len, hidden_size});
    tensor_t normed_hidden_2d_mlp = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);
    ops::rms_norm(normed_hidden_2d_mlp, hidden_states_2d_mlp,
                  weights.mlp_norm_w[layer_idx], meta.epsilon);

    // ========================================
    // 8. SwiGLU MLP
    // ========================================
    tensor_t gate_out = Tensor::create(
        {seq_len, meta.di}, meta.dtype, device_type, 0);
    tensor_t up_out = Tensor::create(
        {seq_len, meta.di}, meta.dtype, device_type, 0);
    tensor_t mlp_output = Tensor::create(
        {seq_len, hidden_size}, meta.dtype, device_type, 0);

    ops::linear(gate_out, normed_hidden_2d_mlp,
                weights.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_out, normed_hidden_2d_mlp,
                weights.mlp_up_w[layer_idx], nullptr);
    ops::swiglu(gate_out, gate_out, up_out);
    ops::linear(mlp_output, gate_out,
                weights.mlp_down_w[layer_idx], nullptr);

    // ========================================
    // 9. Residual Connection (MLP)
    // ========================================
    // Create 3D residual tensor using memcpy for efficient bulk copy
    tensor_t mlp_output_3d = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);
    runtime_api->memcpy_sync(
        mlp_output_3d->data(), mlp_output->data(), residual_bytes, LLAISYS_MEMCPY_H2H);
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
    int64_t *actual_token_ids = is_prefill ? token_ids : token_ids;

    // 1. Embed input tokens
    tensor_t token_indices = Tensor::create(
        {actual_ntoken}, LLAISYS_DTYPE_I64, model->device_type, 0);
    token_indices->load(actual_token_ids);

    // Create 2D hidden_states tensor for embedding (expected by ops::embedding)
    tensor_t hidden_states_2d = Tensor::create(
        {actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);

    // Embedding lookup
    ops::embedding(hidden_states_2d, token_indices, model->weights.in_embed);

    // Reshape to 3D [1, actual_ntoken, model->meta.hs] using memcpy for bulk copy
    tensor_t hidden_states = Tensor::create(
        {1, actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    size_t element_size = llaisys::utils::dsize(model->meta.dtype);
    size_t embed_bytes = actual_ntoken * model->meta.hs * element_size;
    llaisys::core::context().runtime().api()->memcpy_sync(
        hidden_states->data(), hidden_states_2d->data(), embed_bytes, LLAISYS_MEMCPY_H2H);

    // 2. Create position IDs
    tensor_t pos_ids = Tensor::create({1, actual_ntoken}, LLAISYS_DTYPE_I64, model->device_type, 0);
    std::vector<int64_t> pos_data(actual_ntoken);
    for (size_t i = 0; i < actual_ntoken; ++i) {
        pos_data[i] = static_cast<int64_t>(model->current_pos + i);
    }
    pos_ids->load(pos_data.data());

    // 3. Forward pass through all layers
    for (size_t layer_idx = 0; layer_idx < model->meta.nlayer; ++layer_idx) {
        model->forward_layer(layer_idx, hidden_states, pos_ids);
    }

    // 4. Final layer norm
    // View hidden_states as 2D: [1, actual_ntoken, model->meta.hs] -> [actual_ntoken, model->meta.hs] (zero-copy)
    tensor_t final_hidden_states_2d = hidden_states->view({actual_ntoken, model->meta.hs});
    tensor_t normed_output_2d = Tensor::create(
        {actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    ops::rms_norm(normed_output_2d, final_hidden_states_2d,
                  model->weights.out_norm_w, model->meta.epsilon);

    // 5. Project to vocab logits (take last token's output)
    // Use slice to get last token, then view as 2D
    tensor_t normed_output = Tensor::create(
        {1, actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    size_t normed_bytes = actual_ntoken * model->meta.hs * element_size;
    llaisys::core::context().runtime().api()->memcpy_sync(
        normed_output->data(), normed_output_2d->data(), normed_bytes, LLAISYS_MEMCPY_H2H);

    tensor_t last_hidden_slice = normed_output->slice(1, actual_ntoken-1, actual_ntoken);
    tensor_t last_hidden = Tensor::create(
        {1, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    size_t hidden_bytes = model->meta.hs * element_size;
    llaisys::core::context().runtime().api()->memcpy_sync(
        last_hidden->data(), last_hidden_slice->data(), hidden_bytes, LLAISYS_MEMCPY_D2D);

    tensor_t logits = Tensor::create(
        {1, model->meta.voc}, model->meta.dtype, model->device_type, 0);
    ops::linear(logits, last_hidden, model->weights.out_embed, nullptr);

    // 6. Get next token using sampling or argmax
    int64_t next_token;
    tensor_t max_idx_cpu = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    size_t idx_size = llaisys::utils::dsize(LLAISYS_DTYPE_I64);

    if (top_k == 1) {
        // Greedy decoding (argmax)
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, model->device_type, 0);
        tensor_t max_val = Tensor::create({1}, model->meta.dtype, model->device_type, 0);
        ops::argmax(max_idx, max_val, logits);
        llaisys::core::context().runtime().api()->memcpy_sync(
            max_idx_cpu->data(), max_idx->data(), idx_size, LLAISYS_MEMCPY_D2H);
        next_token = *reinterpret_cast<const int64_t *>(max_idx_cpu->data());
    } else {
        // Sampling with top_k and top_p (placeholder: use argmax)
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, model->device_type, 0);
        tensor_t max_val = Tensor::create({1}, model->meta.dtype, model->device_type, 0);
        ops::argmax(max_idx, max_val, logits);
        llaisys::core::context().runtime().api()->memcpy_sync(
            max_idx_cpu->data(), max_idx->data(), idx_size, LLAISYS_MEMCPY_D2H);
        next_token = *reinterpret_cast<const int64_t *>(max_idx_cpu->data());
    }

    // 7. Update position counter
    model->current_pos += actual_ntoken;

    return next_token;
}

void Qwen2ModelResetCache(Qwen2Model *model) {
    if (model) {
        model->reset_cache();
    }
}

} // namespace llaisys::models
