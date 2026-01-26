#pragma once

#include "../../tensor/tensor.hpp"
#include <cmath>

namespace llaisys::models {
struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;   // a.k.a. model.norm.weight
    tensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
    tensor_t *attn_q_w;
    tensor_t *attn_q_b;
    tensor_t *attn_k_w;
    tensor_t *attn_k_b;
    tensor_t *attn_v_w;
    tensor_t *attn_v_b;
    tensor_t *attn_o_w;
    tensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
    tensor_t *mlp_gate_w;
    tensor_t *mlp_up_w;
    tensor_t *mlp_down_w;
};

// Initialize weights with nullptr to avoid uninitialized pointer access
inline Qwen2Weights make_weights_null() {
    Qwen2Weights w;
    w.in_embed = nullptr;
    w.out_embed = nullptr;
    w.out_norm_w = nullptr;
    w.attn_norm_w = nullptr;
    w.attn_q_w = nullptr;
    w.attn_q_b = nullptr;
    w.attn_k_w = nullptr;
    w.attn_k_b = nullptr;
    w.attn_v_w = nullptr;
    w.attn_v_b = nullptr;
    w.attn_o_w = nullptr;
    w.mlp_norm_w = nullptr;
    w.mlp_gate_w = nullptr;
    w.mlp_up_w = nullptr;
    w.mlp_down_w = nullptr;
    return w;
}

struct Qwen2Model {
    Qwen2Meta meta;
    Qwen2Weights weights;

    // KV cache for each layer: shape (maxseq, nkvh, dh)
    tensor_t *k_cache;
    tensor_t *v_cache;

    tensor_t hidden;  // [batch, seq_len, hidden_size]
    tensor_t logits;  // [batch, seq_len, vocab_size]

    // Constructor
    Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);

    // Destructor
    ~Qwen2Model();

    // Forward layer with explicit parameters
    void forward_layer(size_t layer_idx, tensor_t hidden_states, tensor_t pos_ids);

    // Forward
    void forward(tensor_t input_ids, size_t seq_len);

    // Get next token
    int64_t get_next_token(tensor_t logits, size_t pos);

    // Reset KV cache for new sequence
    void reset_cache();

    llaisysDeviceType_t device_type;
    int *device_ids;
    int ndevice;

    size_t current_pos; // Current position in sequence
};

struct Qwen2Model *Qwen2ModelCreate(const Qwen2Meta *meta, llaisysDeviceType_t device_type, int *device_ids, int ndevice);

void Qwen2ModelDestroy(struct Qwen2Model *model);

struct Qwen2Weights *Qwen2ModelWeights(struct Qwen2Model *model);

int64_t Qwen2ModelInfer(struct Qwen2Model *model, int64_t *token_ids, size_t ntoken, int top_k = 1, float top_p = 0.8, float temperature = 0.8);

void Qwen2ModelResetCache(struct Qwen2Model *model);

} // namespace llaisys::models
