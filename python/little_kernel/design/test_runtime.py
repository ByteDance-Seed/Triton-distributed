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
"""
Simple test example for LittleKernel runtime.
This demonstrates how to compile and launch a CUDA kernel from Python.
"""

import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"


@ll_kernel(backend=backend, is_entry=True)
def simple_add_kernel(
    a: ll.Tensor[ll.float32],
    b: ll.Tensor[ll.float32],
    c: ll.Tensor[ll.float32],
    n: ll.int32,
) -> ll.void:
    # Calculate global thread index
    # Note: blockDim.x is accessed directly in CUDA code
    idx = ll.blockIdx_x() * 256 + ll.threadIdx_x()  # Assuming blockDim.x = 256
    if idx < n:
        c[idx] = a[idx] + b[idx]


def test_simple_kernel():
    """Test a simple vector addition kernel."""
    import torch

    # Generate CUDA code
    passes = PASSES[backend]
    code = simple_add_kernel.compile(passes, codegen_cuda)
    print("Generated CUDA code:")
    print("=" * 80)
    print(code)
    print("=" * 80)

    # Build kernel
    n = 1024
    num_threads = 256
    num_blocks = (n + num_threads - 1) // num_threads

    grid = (num_blocks, 1, 1)
    block = (num_threads, 1, 1)

    print(f"\nBuilding kernel with grid={grid}, block={block}...")
    kernel = simple_add_kernel.build(
        passes,
        codegen_cuda,
        grid=grid,
        block=block,
        shared_mem_bytes=0,
        arch="sm_90a",
        verbose=True,
    )

    print("Kernel built successfully!")

    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping kernel launch test")
        return

    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = torch.zeros(n, dtype=torch.float32, device=device)

    print(f"\nLaunching kernel with n={n}...")
    kernel(a, b, c, n)
    kernel.synchronize()

    # Verify result
    expected = a + b
    if torch.allclose(c, expected):
        print("✓ Kernel executed successfully! Result is correct.")
    else:
        print("✗ Kernel result is incorrect!")
        print(f"Max difference: {(c - expected).abs().max().item()}")


if __name__ == "__main__":
    test_simple_kernel()
