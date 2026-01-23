#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2InitWeightArrays(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    __export void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model * model);

    // Weight setters - now take model pointer directly
    __export void llaisysQwen2SetEmbedTokensWeight(struct LlaisysQwen2Model *model, llaisysTensor_t tensor);
    __export void llaisysQwen2SetOutEmbedWeight(struct LlaisysQwen2Model *model, llaisysTensor_t tensor);
    __export void llaisysQwen2SetOutNormWeight(struct LlaisysQwen2Model *model, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnNormWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnQWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnQBias(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnKWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnKBias(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnVWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnVBias(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetAttnOWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetMlpNormWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetMlpGateWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetMlpUpWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);
    __export void llaisysQwen2SetMlpDownWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor);

}
#endif // LLAISYS_MODELS_QWEN2_H
