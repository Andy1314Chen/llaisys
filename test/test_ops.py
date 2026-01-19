import sys
import os
import argparse

# Add ops directory to path
ops_dir = os.path.join(os.path.dirname(__file__), "ops")
sys.path.insert(0, ops_dir)

# Import test functions from individual op test files
from ops.add import test_op_add
from ops.argmax import test_op_argmax
from ops.embedding import test_op_embedding
from ops.linear import test_op_linear
from ops.rms_norm import test_op_rms_norm
from ops.rope import test_op_rope
from ops.self_attention import test_op_self_attention
from ops.swiglu import test_op_swiglu


def test_add():
    print("===Test add===")
    test_op_add((32, 64), "f32", atol=1e-5, rtol=1e-5, device_name="cpu")


def test_argmax():
    print("===Test argmax===")
    test_op_argmax((4096,), "f32", device_name="cpu")


def test_embedding():
    print("===Test embedding===")
    batch_size = 16
    vocab_size = 100
    embed_dim = 64
    test_op_embedding((batch_size,), (vocab_size, embed_dim), "f32", device_name="cpu")


def test_linear():
    print("===Test linear===")
    batch_size = 32
    in_features = 64
    out_features = 128
    test_op_linear(
        (batch_size, out_features),
        (batch_size, in_features),
        (out_features, in_features),
        use_bias=True,
        dtype_name="f32",
        atol=1e-5,
        rtol=1e-5,
        device_name="cpu"
    )


def test_rms_norm():
    print("===Test rms_norm===")
    batch_size = 32
    feature_size = 64
    test_op_rms_norm((batch_size, feature_size), "f32", atol=1e-5, rtol=1e-5, device_name="cpu")


def test_rope():
    print("===Test rope===")
    seq_len = 16
    num_heads = 8
    head_dim = 64
    test_op_rope((seq_len, num_heads, head_dim), (0, seq_len), "f32", atol=1e-4, rtol=1e-4, device_name="cpu")


def test_self_attention():
    print("===Test self_attention===")
    seq_len = 8
    total_seq_len = 16
    num_heads = 4
    kv_num_heads = 2
    head_dim = 32
    test_op_self_attention(seq_len, total_seq_len, num_heads, kv_num_heads, head_dim, "f32", atol=1e-4, rtol=1e-4, device_name="cpu")


def test_swiglu():
    print("===Test swiglu===")
    seq_len = 32
    intermediate_dim = 128
    test_op_swiglu((seq_len, intermediate_dim), "f32", atol=1e-4, rtol=1e-4, device_name="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test llaisys ops")
    parser.add_argument("--ops", type=str, nargs='+', default=None, help="Specify which ops to test")
    args = parser.parse_args()

    test_functions = {
        "add": test_add,
        "argmax": test_argmax,
        "embedding": test_embedding,
        "linear": test_linear,
        "rms_norm": test_rms_norm,
        "rope": test_rope,
        "self_attention": test_self_attention,
        "swiglu": test_swiglu,
    }

    if args.ops:
        for op_name in args.ops:
            if op_name in test_functions:
                test_functions[op_name]()
            else:
                print(f"Warning: Unknown op '{op_name}'")
    else:
        # Run all tests
        test_functions["add"]()
        test_functions["argmax"]()
        test_functions["embedding"]()
        test_functions["linear"]()
        test_functions["rms_norm"]()
        test_functions["rope"]()
        test_functions["self_attention"]()
        test_functions["swiglu"]()

    print("\n\033[92mAll tests passed!\033[0m\n")
