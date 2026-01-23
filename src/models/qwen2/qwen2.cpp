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
    // Debug output
    fprintf(stderr, "[Qwen2Model] Constructor called\n");
    fprintf(stderr, "[Qwen2Model] nlayer=%zu, hs=%zu, nh=%zu, nkvh=%zu, dh=%zu, di=%zu\n",
            meta.nlayer, meta.hs, meta.nh, meta.nkvh, meta.dh, meta.di);
    fprintf(stderr, "[Qwen2Model] maxseq=%zu, voc=%zu, dtype=%d, device_type=%d, device_id=%d\n",
            meta.maxseq, meta.voc, meta.dtype, device_type, did);

    // Validate meta parameters to prevent memory allocation issues
    if (meta.nlayer == 0 || meta.hs == 0 || meta.nh == 0 || meta.nkvh == 0 || 
        meta.dh == 0 || meta.di == 0 || meta.maxseq == 0 || meta.voc == 0) {
        fprintf(stderr, "[Qwen2Model] ERROR: Invalid meta parameters detected\n");
        throw std::invalid_argument("Invalid meta parameters for Qwen2Model");
    }
    
    // Prevent excessively large allocations that could cause memory issues
    if (meta.nlayer > 1000) {
        fprintf(stderr, "[Qwen2Model] ERROR: Too many layers (%zu), maximum allowed is 1000\n", meta.nlayer);
        throw std::invalid_argument("Too many layers for Qwen2Model");
    }

    // Allocate KV cache
    fprintf(stderr, "[Qwen2Model] Allocating KV cache arrays...\n");
    try {
        k_cache = new tensor_t[meta.nlayer];
        v_cache = new tensor_t[meta.nlayer];
    } catch (const std::bad_alloc& e) {
        fprintf(stderr, "[Qwen2Model] ERROR: Failed to allocate KV cache arrays: %s\n", e.what());
        throw;
    }
    fprintf(stderr, "[Qwen2Model] KV cache arrays allocated\n");

    // Initialize all tensor pointers to nullptr first to prevent undefined behavior
    for (size_t i = 0; i < meta.nlayer; ++i) {
        k_cache[i] = nullptr;
        v_cache[i] = nullptr;
    }

    for (size_t i = 0; i < meta.nlayer; ++i) {
        fprintf(stderr, "[Qwen2Model] Creating KV cache tensors for layer %zu...\n", i);
        try {
            // K cache: [maxseq, nkvh, dh]
            k_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh},
                                        meta.dtype, device_type, did);
            if (!k_cache[i]) {
                fprintf(stderr, "[Qwen2Model] ERROR: Failed to create K cache tensor for layer %zu\n", i);
                throw std::runtime_error("Failed to create K cache tensor");
            }
            
            // V cache: [maxseq, nkvh, dh]
            v_cache[i] = Tensor::create({meta.maxseq, meta.nkvh, meta.dh},
                                        meta.dtype, device_type, did);
            if (!v_cache[i]) {
                fprintf(stderr, "[Qwen2Model] ERROR: Failed to create V cache tensor for layer %zu\n", i);
                throw std::runtime_error("Failed to create V cache tensor");
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "[Qwen2Model] ERROR creating cache tensor for layer %zu: %s\n", i, e.what());
            throw;
        }
    }
    fprintf(stderr, "[Qwen2Model] KV cache tensors created\n");

    // hidden: [1, maxseq, hidden_size]
    fprintf(stderr, "[Qwen2Model] Creating hidden tensor...\n");
    try {
        hidden = Tensor::create({1, meta.maxseq, meta.hs}, meta.dtype, device_type, did);
        if (!hidden) {
            fprintf(stderr, "[Qwen2Model] ERROR: Failed to create hidden tensor\n");
            throw std::runtime_error("Failed to create hidden tensor");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[Qwen2Model] ERROR creating hidden tensor: %s\n", e.what());
        throw;
    }
    fprintf(stderr, "[Qwen2Model] hidden tensor created\n");

    // logits: [1, vocab_size] - for single token output
    fprintf(stderr, "[Qwen2Model] Creating logits tensor...\n");
    try {
        logits = Tensor::create({1, meta.voc}, meta.dtype, device_type, did);
        if (!logits) {
            fprintf(stderr, "[Qwen2Model] ERROR: Failed to create logits tensor\n");
            throw std::runtime_error("Failed to create logits tensor");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[Qwen2Model] ERROR creating logits tensor: %s\n", e.what());
        throw;
    }
    fprintf(stderr, "[Qwen2Model] logits tensor created\n");

    fprintf(stderr, "[Qwen2Model] Constructor completed successfully\n");
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
        {1, seq_len, meta.di}, meta.dtype, device_type, 0);

    tensor_t up_out = Tensor::create(
        {1, seq_len, meta.di}, meta.dtype, device_type, 0);

    tensor_t mlp_output = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);

    tensor_t residual = Tensor::create(
        {1, seq_len, hidden_size}, meta.dtype, device_type, 0);

    // ========================================
    // 1. RMSNorm (input layernorm)
    // ========================================
    ops::rms_norm(normed_hidden, hidden_states,
                  weights.attn_norm_w[layer_idx], meta.epsilon);

    // ========================================
    // 2. QKV Projection
    // ========================================
    // QKV linear projections need input shape [1, seq_len, hidden_size] and output shape [1, seq_len, num_heads*head_dim]
    // So we need temporary tensors for projection results, then reshape to [seq_len, num_heads, head_dim]
    tensor_t q_flat = Tensor::create(
        {1, seq_len, num_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t k_flat = Tensor::create(
        {1, seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);
    tensor_t v_flat = Tensor::create(
        {1, seq_len, num_kv_heads * head_dim}, meta.dtype, device_type, 0);

    ops::linear(q_flat, normed_hidden,
                weights.attn_q_w[layer_idx], weights.attn_q_b[layer_idx]);
    ops::linear(k_flat, normed_hidden,
                weights.attn_k_w[layer_idx], weights.attn_k_b[layer_idx]);
    ops::linear(v_flat, normed_hidden,
                weights.attn_v_w[layer_idx], weights.attn_v_b[layer_idx]);

    // Copy flat results to reshaped tensors (no actual reshape needed - just reinterpret memory)
    // q_flat: [1, seq_len, num_heads*head_dim] -> q: [seq_len, num_heads, head_dim]
    // k_flat: [1, seq_len, num_kv_heads*head_dim] -> k: [seq_len, num_kv_heads, head_dim]
    // v_flat: [1, seq_len, num_kv_heads*head_dim] -> v: [seq_len, num_kv_heads, head_dim]
    // Note: Since reshape() is not implemented, we'll use memcpy to copy the data
    size_t q_size = seq_len * num_heads * head_dim * llaisys::utils::dsize(meta.dtype);
    size_t kv_size = seq_len * num_kv_heads * head_dim * llaisys::utils::dsize(meta.dtype);
    llaisys::core::context().runtime().api()->memcpy_sync(
        q->data(), q_flat->data(), q_size, LLAISYS_MEMCPY_D2D);
    llaisys::core::context().runtime().api()->memcpy_sync(
        k->data(), k_flat->data(), kv_size, LLAISYS_MEMCPY_D2D);
    llaisys::core::context().runtime().api()->memcpy_sync(
        v->data(), v_flat->data(), kv_size, LLAISYS_MEMCPY_D2D);

    // ========================================
    // 3. RoPE (Rotary Position Embedding)
    // ========================================
    ops::rope(q_rope, q, pos_ids, meta.theta);
    ops::rope(k_rope, k, pos_ids, meta.theta);

    // ========================================
    // 4. Update KV Cache
    // ========================================
    // For generation mode (single token), update cache at current position
    // For prefill mode (multiple tokens), update cache from current_pos to current_pos+seq_len
    for (size_t i = 0; i < seq_len; ++i) {
        size_t cache_pos = current_pos + i;
        if (cache_pos < meta.maxseq) {
            // Get pointers to cache tensors
            tensor_t k_cache_tensor = k_cache[layer_idx];
            tensor_t v_cache_tensor = v_cache[layer_idx];

            // Get source and destination pointers
            // k_rope: [seq_len, nkvh, dh] -> slice along dim=0 to get position i
            tensor_t k_slice = k_rope->slice(0, i, i+1); // [1, nkvh, dh]
            tensor_t k_cache_slice = k_cache_tensor->slice(0, cache_pos, cache_pos+1); // [1, nkvh, dh]

            // v: [seq_len, nkvh, dh] -> slice along dim=0 to get position i
            tensor_t v_slice = v->slice(0, i, i+1); // [1, nkvh, dh]
            tensor_t v_cache_slice = v_cache_tensor->slice(0, cache_pos, cache_pos+1); // [1, nkvh, dh]

            // Copy data using memcpy
            size_t cache_size = meta.nkvh * meta.dh * llaisys::utils::dsize(meta.dtype);
            llaisys::core::context().runtime().api()->memcpy_sync(
                k_cache_slice->data(), k_slice->data(), cache_size, LLAISYS_MEMCPY_D2D);
            llaisys::core::context().runtime().api()->memcpy_sync(
                v_cache_slice->data(), v_slice->data(), cache_size, LLAISYS_MEMCPY_D2D);
        }
    }

    // ========================================
    // 5. Self-Attention
    // ========================================
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // For generation, use cached K and V for all previous positions
    // We need to concatenate cached K/V with current K/V
    // k_cache[layer_idx]: [maxseq, nkvh, dh] - use [0:current_pos+seq_len, nkvh, dh]
    // v_cache[layer_idx]: [maxseq, nkvh, dh] - use [0:current_pos+seq_len, nkvh, dh]
    tensor_t k_cached = k_cache[layer_idx]->slice(0, 0, current_pos + seq_len); // [current_pos+seq_len, nkvh, dh]
    tensor_t v_cached = v_cache[layer_idx]->slice(0, 0, current_pos + seq_len); // [current_pos+seq_len, nkvh, dh]

    // Attention: [seq_len, num_heads, head_dim]
    // q_rope: [seq_len, num_heads, head_dim] - already in correct format
    // k_cached: [current_pos+seq_len, nkvh, dh] - already in correct format
    // v_cached: [current_pos+seq_len, nkvh, dh] - already in correct format
    // attn_output: [seq_len, num_heads, head_dim] - already in correct format
    ops::self_attention(attn_output, q_rope, k_cached, v_cached, scale);

    // ========================================
    // 6. Output Projection
    // ========================================
    // attn_output: [seq_len, num_heads, head_dim] - need to flatten to [1, seq_len, num_heads*head_dim]
    // Create temporary tensor for linear operation
    tensor_t attn_output_flat = Tensor::create(
        {1, seq_len, num_heads * head_dim}, meta.dtype, device_type, 0);
    size_t attn_size = seq_len * num_heads * head_dim * llaisys::utils::dsize(meta.dtype);
    llaisys::core::context().runtime().api()->memcpy_sync(
        attn_output_flat->data(), attn_output->data(), attn_size, LLAISYS_MEMCPY_D2D);

    ops::linear(attn_output_flat, attn_output_flat,
                weights.attn_o_w[layer_idx], nullptr);

    // ========================================
    // 7. Residual Connection (Attention)
    // ========================================
    residual = hidden_states; // Copy original hidden states
    ops::add(hidden_states, residual, attn_output_flat);

    // ========================================
    // 8. RMSNorm (post attention layernorm)
    // ========================================
    ops::rms_norm(normed_hidden, hidden_states,
                  weights.mlp_norm_w[layer_idx], meta.epsilon);

    // ========================================
    // 9. SwiGLU MLP
    // ========================================
    ops::linear(gate_out, normed_hidden,
                weights.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_out, normed_hidden,
                weights.mlp_up_w[layer_idx], nullptr);

    // Compute SwiGLU activation
    tensor_t swiglu_intermediate = Tensor::create(
        {1, seq_len, meta.di}, meta.dtype, device_type, 0);
    ops::swiglu(swiglu_intermediate, gate_out, up_out);

    // Apply down projection
    ops::linear(mlp_output, swiglu_intermediate,
                weights.mlp_down_w[layer_idx], nullptr);

    // ========================================
    // 10. Residual Connection (MLP)
    // ========================================
    residual = hidden_states;
    ops::add(hidden_states, residual, mlp_output);

    // Note: current_pos is updated by the caller (Qwen2ModelInfer) after all layers are processed
}

void Qwen2Model::forward(tensor_t input_ids, size_t seq_len) {
    // 1. Embedding lookup - create temporary tensor for hidden states
    tensor_t temp_hidden = Tensor::create(
        {1, seq_len, meta.hs}, meta.dtype, device_type, 0);
    ops::embedding(temp_hidden, input_ids, weights.in_embed);

    // 2. Create position IDs
    tensor_t pos_ids = Tensor::create({1, seq_len}, LLAISYS_DTYPE_I64, device_type, 0);
    // Fill with [current_pos, current_pos+1, ..., current_pos+seq_len-1]
    // Implementation depends on tensor fill capability

    // 3. Transformer Layers
    for (size_t layer_idx = 0; layer_idx < meta.nlayer; layer_idx++) {
        forward_layer(layer_idx, temp_hidden, pos_ids);
    }

    // 4. Final Norm
    tensor_t final_hidden = Tensor::create({1, seq_len, meta.hs}, meta.dtype, device_type, 0);
    ops::rms_norm(final_hidden, temp_hidden, weights.out_norm_w, meta.epsilon);

    // 5. Output Projection to vocabulary (take last token's output)
    // Use the correct slice method: slice(dim, start, end)
    tensor_t last_hidden = final_hidden->slice(1, seq_len-1, seq_len); // Get last token along sequence dimension
    ops::linear(logits, last_hidden, weights.out_embed, nullptr);
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

int64_t Qwen2ModelInfer(Qwen2Model *model, int64_t *token_ids, size_t ntoken) {
    // Detect prefill vs decode mode
    bool is_prefill = (model->current_pos == 0);

    // For decode mode, only process the last token
    size_t actual_ntoken = is_prefill ? ntoken : 1;
    int64_t *actual_token_ids = is_prefill ? token_ids : (token_ids + ntoken - 1);

    // 1. Embed input tokens
    tensor_t token_indices = Tensor::create(
        {1, actual_ntoken}, LLAISYS_DTYPE_I64, model->device_type, 0);
    token_indices->load(actual_token_ids);

    tensor_t hidden_states = Tensor::create(
        {1, actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);

    // Embedding lookup: hidden_states[token] = in_embed[token]
    ops::embedding(hidden_states, token_indices, model->weights.in_embed);

    // 2. Create position IDs [current_pos, current_pos+1, ..., current_pos+actual_ntoken-1]
    tensor_t pos_ids = Tensor::create(
        {1, actual_ntoken}, LLAISYS_DTYPE_I64, model->device_type, 0);
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
    tensor_t normed_output = Tensor::create(
        {1, actual_ntoken, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    ops::rms_norm(normed_output, hidden_states,
                  model->weights.out_norm_w, model->meta.epsilon);

    // 5. Project to vocab logits (take last token's output)
    // Use the correct slice method: slice(dim, start, end)
    tensor_t last_hidden_slice = normed_output->slice(1, actual_ntoken-1, actual_ntoken); // Get last token along sequence dimension
    
    // The slice returns [1, 1, 1536], but Linear expects [1, 1536]
    // We need to squeeze the sequence dimension
    // Since the slice is not contiguous, we can't simply reshape
    // Instead, create a new contiguous tensor and copy the data
    tensor_t last_hidden = Tensor::create(
        {1, model->meta.hs}, model->meta.dtype, model->device_type, 0);
    
    // Copy data from the sliced tensor to the new contiguous tensor
    // last_hidden_slice is [1, 1, 1536], last_hidden is [1, 1536]
    size_t copy_size = model->meta.hs * llaisys::utils::dsize(model->meta.dtype);
    llaisys::core::context().runtime().api()->memcpy_sync(
        last_hidden->data(), last_hidden_slice->data(), copy_size, LLAISYS_MEMCPY_H2H);
    
    tensor_t logits = Tensor::create(
        {1, model->meta.voc}, model->meta.dtype, model->device_type, 0);
    ops::linear(logits, last_hidden, model->weights.out_embed, nullptr);

    // 6. Argmax to get next token
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, model->device_type, 0);
    tensor_t max_val = Tensor::create({1}, model->meta.dtype, model->device_type, 0);

    ops::argmax(max_idx, max_val, logits);

    // 7. Read the token ID
    const std::byte *data_ptr = max_idx->data();
    int64_t next_token = *reinterpret_cast<const int64_t *>(data_ptr);

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
