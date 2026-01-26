#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"
#include <cstring>
#include <cstdio>
#include <exception>

__C {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const struct LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    // Convert C meta to C++ meta
    llaisys::models::Qwen2Meta cpp_meta;
    cpp_meta.dtype = meta->dtype;
    cpp_meta.nlayer = meta->nlayer;
    cpp_meta.hs = meta->hs;
    cpp_meta.nh = meta->nh;
    cpp_meta.nkvh = meta->nkvh;
    cpp_meta.dh = meta->dh;
    cpp_meta.di = meta->di;
    cpp_meta.maxseq = meta->maxseq;
    cpp_meta.voc = meta->voc;
    cpp_meta.epsilon = meta->epsilon;
    cpp_meta.theta = meta->theta;
    cpp_meta.end_token = meta->end_token;

    auto *cpp_model = llaisys::models::Qwen2ModelCreate(&cpp_meta, device, device_ids, ndevice);

    // Return as void* - C struct is just a handle, not a real struct
    return reinterpret_cast<struct LlaisysQwen2Model*>(cpp_model);
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    llaisys::models::Qwen2ModelDestroy(cpp_model);
}

// Helper function to initialize weight arrays
__export void llaisysQwen2InitWeightArrays(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    auto *cpp_model = static_cast<llaisys::models::Qwen2Model*>(reinterpret_cast<void*>(model));
    
    // Check if we can safely access the meta data by validating nlayer bounds first
    size_t nlayer = 0;
    try {
        // volatile auto device_type = cpp_model->device_type;
        // Access nlayer with extra safety checks
        nlayer = cpp_model->meta.nlayer;
    } catch (...) {
        return;
    }
    
    // Make sure nlayer is reasonable
    if (nlayer == 0 || nlayer > 10000) {  // Sanity check - no realistic model would have 10k+ layers
        return;
    }
    
    // Initialize all weight arrays and set elements to nullptr
    try {
        cpp_model->weights.attn_norm_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_norm_w[i] = nullptr;
        }

        cpp_model->weights.attn_q_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_q_w[i] = nullptr;
        }

        cpp_model->weights.attn_q_b = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_q_b[i] = nullptr;
        }

        cpp_model->weights.attn_k_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_k_w[i] = nullptr;
        }

        cpp_model->weights.attn_k_b = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_k_b[i] = nullptr;
        }

        cpp_model->weights.attn_v_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_v_w[i] = nullptr;
        }

        cpp_model->weights.attn_v_b = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_v_b[i] = nullptr;
        }

        cpp_model->weights.attn_o_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.attn_o_w[i] = nullptr;
        }

        cpp_model->weights.mlp_norm_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.mlp_norm_w[i] = nullptr;
        }

        cpp_model->weights.mlp_gate_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.mlp_gate_w[i] = nullptr;
        }

        cpp_model->weights.mlp_up_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.mlp_up_w[i] = nullptr;
        }

        cpp_model->weights.mlp_down_w = new llaisys::tensor_t[nlayer];
        for (size_t i = 0; i < nlayer; ++i) {
            cpp_model->weights.mlp_down_w[i] = nullptr;
        }
    } catch (const std::bad_alloc&) {
        // Clean up any partial allocations
        // For now, just return - in production code you'd want more sophisticated cleanup
        return;
    } catch (...) {
        return;
    }
}

// Functions to set individual weights - now take model pointer directly
__export void llaisysQwen2SetEmbedTokensWeight(struct LlaisysQwen2Model *model, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->in_embed = tensor->tensor;
}

__export void llaisysQwen2SetOutEmbedWeight(struct LlaisysQwen2Model *model, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->out_embed = tensor->tensor;
}

__export void llaisysQwen2SetOutNormWeight(struct LlaisysQwen2Model *model, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->out_norm_w = tensor->tensor;
}

// Functions to set layer weights by index
__export void llaisysQwen2SetAttnNormWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->attn_norm_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetAttnQWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->attn_q_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetAttnQBias(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    if (cpp_weights->attn_q_b) {
        cpp_weights->attn_q_b[layer_idx] = tensor->tensor;
    }
}

__export void llaisysQwen2SetAttnKWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->attn_k_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetAttnKBias(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    if (cpp_weights->attn_k_b) {
        cpp_weights->attn_k_b[layer_idx] = tensor->tensor;
    }
}

__export void llaisysQwen2SetAttnVWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->attn_v_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetAttnVBias(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    if (cpp_weights->attn_v_b) {
        cpp_weights->attn_v_b[layer_idx] = tensor->tensor;
    }
}

__export void llaisysQwen2SetAttnOWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->attn_o_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetMlpNormWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->mlp_norm_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetMlpGateWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->mlp_gate_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetMlpUpWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->mlp_up_w[layer_idx] = tensor->tensor;
}

__export void llaisysQwen2SetMlpDownWeight(struct LlaisysQwen2Model *model, size_t layer_idx, llaisysTensor_t tensor) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    auto *cpp_weights = llaisys::models::Qwen2ModelWeights(cpp_model);
    cpp_weights->mlp_down_w[layer_idx] = tensor->tensor;
}

// Inference function
__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, int top_k, float top_p, float temperature) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    int64_t result = llaisys::models::Qwen2ModelInfer(cpp_model, token_ids, ntoken, top_k, top_p, temperature);
    return result;
}

// Reset cache function
__export void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model) {
    auto *cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
    llaisys::models::Qwen2ModelResetCache(cpp_model);
}

} // __C
