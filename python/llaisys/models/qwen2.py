from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType
from ..tensor import Tensor
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
                           (hasattr(np, 'bfloat16') and tensor.dtype == np.bfloat16):
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
        print(f"Loaded {len(self.weights_data)} weights from safetensors.")

        # Parse model config
        print("Parse model config...")
        self._parse_config(model_path)
        
        # Create model
        print("Create model...")
        self._create_model()

    def _parse_config(self, model_path):
        # Read configuration from config.json
        import json
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

    def _create_model(self):
        from ..libllaisys.models import LlaisysQwen2Meta  # Import the correct Qwen2Meta

        meta = LlaisysQwen2Meta()
        meta.dtype = 13  # LLAISYS_DTYPE_F32 (assuming this is the correct value)
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

        # IMPORTANT: Pass meta by reference (byref) since C API expects POINTER(LlaisysQwen2Meta)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),  # Use byref to pass pointer
            self.device.value,
            None,
            1
        )

        # Load weights into the C++ model
        self._load_weights_to_cpp()
        
        print(f"[Python] self._model value: {self._model}")
        print(f"[Python] self._model type: {type(self._model)}")

    def _load_weights_to_cpp(self):
        """Load weights from safetensors into C++ model"""
        from ..tensor import Tensor
        import ctypes

        print("[Python] _load_weights_to_cpp started")
        # Initialize weight arrays in C++
        print("[Python] Calling llaisysQwen2InitWeightArrays...")
        LIB_LLAISYS.llaisysQwen2InitWeightArrays(self._model)
        print("[Python] Weight arrays initialized")

        # Process each loaded weight
        count = 0
        for name, tensor_data in self.weights_data.items():
            count += 1
            if count % 10 == 0 or count == len(self.weights_data):
                print(f"[Python] Processing weight {count}/{len(self.weights_data)}: {name[:50]}...")

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
                print("[Python] Setting embed_tokens weight...")
                LIB_LLAISYS.llaisysQwen2SetEmbedTokensWeight(self._model, weight_tensor.lib_tensor())
            elif name == 'model.norm.weight':
                print("[Python] Setting out_norm weight...")
                LIB_LLAISYS.llaisysQwen2SetOutNormWeight(self._model, weight_tensor.lib_tensor())
            elif name == 'lm_head.weight':
                print("[Python] Setting out_embed weight...")
                LIB_LLAISYS.llaisysQwen2SetOutEmbedWeight(self._model, weight_tensor.lib_tensor())

        print("[Python] _load_weights_to_cpp completed")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        import ctypes

        # Reset cache before starting a new generation
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

        input_tokens = list(inputs)

        output_tokens = []
        for _ in range(max_new_tokens):
            # Convert input_tokens to ctypes array
            token_array = (ctypes.c_longlong * len(input_tokens))(*input_tokens)
            
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                len(input_tokens)
            )

            if top_k == 1:
                sampled_token = next_token
            else:
                sampled_token = next_token

            if sampled_token == 151643:  # End token
                break

            output_tokens.append(sampled_token)
            input_tokens.append(sampled_token)

        return output_tokens

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
