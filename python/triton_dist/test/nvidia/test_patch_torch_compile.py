################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import torch
import triton
from triton_dist.tools import apply_triton340_inductor_patch


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def to_be_compiled(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def main():
    from packaging.version import Version

    torch_version = torch.__version__
    triton_version = triton.__version__
    require_patched_torch = (Version("2.7.0") <= Version(torch_version) < Version("2.8.1"))

    if not require_patched_torch:
        print("🅾️ Skiping test for monkey patch inductor: "
              f"torch {torch_version} with triton {triton_version}")
        return

    torch.cuda.manual_seed(42)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    B, H, T, D = 4, 32, 4096, 128
    q = torch.randn([B, H, T, D])
    k = torch.randn([B, H, T, D])
    cos = torch.randn([B, T, D])
    sin = torch.randn([B, T, D])
    inps = (q, k, cos, sin)

    print(f">>> torch {torch_version}, triton {triton_version}")
    print("🔧 torch.compile WITHOUT patch ...")
    try:
        compiled_func = torch.compile(to_be_compiled)
        compiled_func(*inps)
    except Exception as e:
        print(f"❌ Compilation failed with error:\n  {e}")
        raise e

    print("🔧 torch.compile WITH patch ...")
    apply_triton340_inductor_patch()
    compiled_func = torch.compile(to_be_compiled)
    compiled_func(*inps)
    print("✅ Compilation pass with patch applied")


if __name__ == "__main__":
    main()
