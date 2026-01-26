from ctypes import POINTER, c_int, c_size_t, c_float, c_int64, c_void_p, c_uint64
from .llaisys_types import llaisysDeviceType_t
import ctypes


# Use c_size_t to handle 64-bit pointers (size_t is 8 bytes on 64-bit)
llaisysQwen2Model_t = c_void_p


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


def load_models(lib):
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreate.restype = c_void_p
    
    # Helper function to ensure returned pointer is c_void_p
    def _ensure_c_void_p(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, int):
                return c_void_p(result)
            return result
        return wrapper
    
    lib.llaisysQwen2ModelCreate = _ensure_c_void_p(lib.llaisysQwen2ModelCreate)

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2InitWeightArrays
    lib.llaisysQwen2InitWeightArrays.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2InitWeightArrays.restype = None

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t, POINTER(c_int64), c_size_t, c_int, c_float, c_float]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    # llaisysQwen2ModelResetCache
    lib.llaisysQwen2ModelResetCache.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetCache.restype = None

    # Weight setters - now take model pointer directly
    lib.llaisysQwen2SetEmbedTokensWeight.argtypes = [llaisysQwen2Model_t, c_void_p]
    lib.llaisysQwen2SetEmbedTokensWeight.restype = None

    lib.llaisysQwen2SetOutEmbedWeight.argtypes = [llaisysQwen2Model_t, c_void_p]
    lib.llaisysQwen2SetOutEmbedWeight.restype = None

    lib.llaisysQwen2SetOutNormWeight.argtypes = [llaisysQwen2Model_t, c_void_p]
    lib.llaisysQwen2SetOutNormWeight.restype = None

    lib.llaisysQwen2SetAttnNormWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnNormWeight.restype = None

    lib.llaisysQwen2SetAttnQWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnQWeight.restype = None

    lib.llaisysQwen2SetAttnQBias.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnQBias.restype = None

    lib.llaisysQwen2SetAttnKWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnKWeight.restype = None

    lib.llaisysQwen2SetAttnKBias.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnKBias.restype = None

    lib.llaisysQwen2SetAttnVWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnVWeight.restype = None

    lib.llaisysQwen2SetAttnVBias.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnVBias.restype = None

    lib.llaisysQwen2SetAttnOWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetAttnOWeight.restype = None

    lib.llaisysQwen2SetMlpNormWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetMlpNormWeight.restype = None

    lib.llaisysQwen2SetMlpGateWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetMlpGateWeight.restype = None

    lib.llaisysQwen2SetMlpUpWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetMlpUpWeight.restype = None

    lib.llaisysQwen2SetMlpDownWeight.argtypes = [llaisysQwen2Model_t, c_size_t, c_void_p]
    lib.llaisysQwen2SetMlpDownWeight.restype = None