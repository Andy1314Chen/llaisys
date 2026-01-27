from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType
from ..tensor import Tensor
import ml_dtypes
from pathlib import Path
import ctypes
import safetensors
import numpy as np


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self.device = device

        # Load model weights
        self.weights_data = {}
        for file in sorted(model_path.glob("*.safetensors")):
            # First try to open with numpy framework
            try:
                with safetensors.safe_open(file, framework="numpy", device="cpu") as f:
                    for name in f.keys():
                        tensor = f.get_tensor(name)
                        # Convert bfloat16 to float32 if needed
                        # Since numpy doesn't have native bfloat16 support in some versions,
                        # we detect bfloat16 differently
                        if hasattr(tensor, 'dtype') and str(tensor.dtype) in ['bfloat16', '<V2'] or \
                           tensor.dtype == ml_dtypes.bfloat16:
                            # Convert to float32 for compatibility
                            tensor = tensor.astype(np.float32)
                        self.weights_data[name] = tensor
            except TypeError as e:
                if 'bfloat16' in str(e):
                    # If numpy framework fails due to bfloat16, try with pt (PyTorch) framework and convert
                    import torch
                    with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                        for name in f.keys():
                            tensor = f.get_tensor(name)
                            # Convert PyTorch tensor to NumPy, which handles bfloat16 conversion
                            if tensor.dtype == torch.bfloat16:
                                tensor = tensor.to(torch.float32)
                            numpy_tensor = tensor.detach().cpu().numpy()
                            self.weights_data[name] = numpy_tensor
                else:
                    raise
        # Model weights loaded from safetensors

        # Parse model config
        self._parse_config(model_path)
        
        # Create model
        self._create_model()

    def _parse_config(self, model_path):
        # Read configuration from config.json
        import json
        from ..libllaisys import DataType
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)
        self.vocab_size = config["vocab_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config["rope_theta"]
        self.intermediate_size = config["intermediate_size"]
        self.end_token = config.get("eos_token_id", 151643)

        # Parse torch_dtype from config
        torch_dtype_str = config.get("torch_dtype", "float32")
        dtype_map = {
            "float32": DataType.F32,
            "float16": DataType.F32,
            "bfloat16": DataType.F32
        }
        self.dtype = dtype_map.get(torch_dtype_str, DataType.F32)

    def _create_model(self):
        from ..libllaisys.models import LlaisysQwen2Meta  # Import the correct Qwen2Meta

        meta = LlaisysQwen2Meta()
        meta.dtype = self.dtype.value
        meta.nlayer = self.num_layers
        meta.hs = self.hidden_size
        meta.nh = self.num_heads
        meta.nkvh = self.num_key_value_heads
        meta.dh = self.hidden_size // self.num_heads
        meta.di = self.intermediate_size  # Use config value instead of hardcoded
        # TEMPORARY: Reduce maxseq for testing to avoid OOM
        # The KV cache requires too much memory: 131072 * 2 * 128 * 28 * 4 bytes = 3.5GB
        # Using a smaller maxseq for now - this limits the maximum sequence length
        # TODO: Implement KV cache that doesn't pre-allocate all tensors
        meta.maxseq = min(self.max_position_embeddings, 2048)  # Reduced for testing
        meta.voc = self.vocab_size
        meta.epsilon = self.rms_norm_eps
        meta.theta = self.rope_theta
        meta.end_token = self.end_token

        # Save meta for later reference
        self.meta = meta

        # IMPORTANT: Pass meta by reference (byref) since C API expects POINTER(LlaisysQwen2Meta)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),  # Use byref to pass pointer
            self.device.value,
            None,
            1
        )

        # Load weights into the C++ model
        self._load_weights_to_cpp()


    def _load_weights_to_cpp(self):
        """Load weights from safetensors into C++ model"""
        from ..tensor import Tensor
        import ctypes

        # Initialize weight arrays in C++
        LIB_LLAISYS.llaisysQwen2InitWeightArrays(self._model)

        # Process each loaded weight
        for name, tensor_data in self.weights_data.items():

            weight_tensor = Tensor.from_numpy(tensor_data)

            # Handle attention norm weights (input_layernorm)
            if name.endswith('.input_layernorm.weight'):
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnNormWeight(self._model, layer_idx, weight_tensor.lib_tensor())

            # Handle attention weights
            elif '.self_attn.q_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnQWeight(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.self_attn.k_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnKWeight(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.self_attn.v_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnVWeight(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.self_attn.o_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnOWeight(self._model, layer_idx, weight_tensor.lib_tensor())

            # Handle attention biases
            elif '.self_attn.q_proj.bias' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnQBias(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.self_attn.k_proj.bias' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnKBias(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.self_attn.v_proj.bias' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetAttnVBias(self._model, layer_idx, weight_tensor.lib_tensor())

            # Handle MLP norm weights (post_attention_layernorm)
            elif name.endswith('.post_attention_layernorm.weight'):
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetMlpNormWeight(self._model, layer_idx, weight_tensor.lib_tensor())

            # Handle MLP weights
            elif '.mlp.gate_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetMlpGateWeight(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.mlp.up_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetMlpUpWeight(self._model, layer_idx, weight_tensor.lib_tensor())
            elif '.mlp.down_proj.weight' in name:
                layer_idx = int(name.split('.')[2])
                LIB_LLAISYS.llaisysQwen2SetMlpDownWeight(self._model, layer_idx, weight_tensor.lib_tensor())

            # Handle common weights
            elif name == 'model.embed_tokens.weight':
                LIB_LLAISYS.llaisysQwen2SetEmbedTokensWeight(self._model, weight_tensor.lib_tensor())
            elif name == 'model.norm.weight':
                LIB_LLAISYS.llaisysQwen2SetOutNormWeight(self._model, weight_tensor.lib_tensor())
            elif name == 'lm_head.weight':
                LIB_LLAISYS.llaisysQwen2SetOutEmbedWeight(self._model, weight_tensor.lib_tensor())

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        import ctypes

        # Only reset cache if this is the beginning of a new generation sequence
        # We detect this by checking if inputs contains more than one token
        # For continuation (decode mode), inputs should contain only the last token
        input_tokens = list(inputs)
        
        # Reset cache only for new sequences (when we have multiple input tokens)
        if len(input_tokens) > 1:
            LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

        output_tokens = []

        # Prefill phase: Process all input tokens at once
        # Save the input tokens - they should be included in the final output
        # because we need to return the complete sequence (input + generated tokens)
        # to match PyTorch's behavior
        input_tokens_copy = list(input_tokens)
        if input_tokens:
            token_array = (ctypes.c_longlong * len(input_tokens))(*input_tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                len(input_tokens),
                top_k,
                top_p,
                temperature
            )
            if next_token == self.meta.end_token:  # End token
                return input_tokens_copy + []  # Return input tokens only
            output_tokens.append(next_token)
            last_token = next_token

        # Decode phase: Generate one token at a time
        # In decode mode, we only pass the last generated token
        while len(output_tokens) < max_new_tokens:
            # Pass only the last token for decode mode
            # The C++ code will use token_ids[ntoken-1] which is the last token
            token_array = (ctypes.c_longlong * 1)(*([last_token]))

            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                1,  # ntoken = 1 for decode mode
                top_k,
                top_p,
                temperature
            )
            
            
            if next_token == self.meta.end_token:  # End token
                output_tokens.append(next_token)
                break
            
            output_tokens.append(next_token)
            last_token = next_token

        # Return input_tokens + generated_tokens to match PyTorch behavior
        # PyTorch's generate() returns the complete sequence including input tokens
        return input_tokens_copy + output_tokens

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
