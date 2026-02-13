# LittleKernel Runtime

The LittleKernel runtime provides compilation and execution of CUDA kernels, similar to Triton's runtime.

## Features

1. **Compile CUDA code**: Compile CUDA source to CUBIN using nvcc
2. **Load kernels**: Load compiled kernels via the CUDA Driver API
3. **Launch kernels**: Pass data from Python and launch kernels on the GPU

## Usage

### Basic Usage

```python
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"

@ll_kernel(backend=backend, is_entry=True)
def my_kernel(
    a: ll.Tensor[ll.float32],
    b: ll.Tensor[ll.float32],
    n: ll.int32,
) -> ll.void:
    idx = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if idx < n:
        b[idx] = a[idx] * 2.0

# Compile and build the kernel
passes = PASSES[backend]
kernel = my_kernel.build(
    passes,
    codegen_cuda,
    grid=(num_blocks, 1, 1),      # Grid dimensions
    block=(256, 1, 1),             # Block dimensions
    shared_mem_bytes=0,            # Dynamic shared memory size (bytes)
    arch="sm_90a",                 # Target architecture
    verbose=True,                   # Print compilation info
)

# Launch the kernel
import torch
a = torch.randn(1024, dtype=torch.float32, device="cuda")
b = torch.zeros(1024, dtype=torch.float32, device="cuda")
n = 1024

kernel(a, b, n)
kernel.synchronize()
```

### Advanced Usage

#### Using Cache

```python
kernel = my_kernel.build(
    passes,
    codegen_cuda,
    grid=(num_blocks, 1, 1),
    block=(256, 1, 1),
    cache_dir="./kernel_cache",  # Cache directory
)
```

#### Using a Custom Stream

```python
stream = torch.cuda.Stream()
kernel(a, b, n, stream=stream)
kernel.synchronize(stream=stream)
```

## API Reference

### `LLKernel.build()`

Compile and build a launchable kernel.

**Parameters:**
- `passes`: List of compiler passes
- `codegen_func`: Code generation function (e.g., `codegen_cuda`)
- `grid`: Grid dimensions `(gridDimX, gridDimY, gridDimZ)`
- `block`: Block dimensions `(blockDimX, blockDimY, blockDimZ)`
- `shared_mem_bytes`: Dynamic shared memory size in bytes
- `arch`: Target architecture (e.g., `"sm_90a"`); auto-detected if `None`
- `cache_dir`: Cache directory; no caching if `None`
- `verbose`: Whether to print compilation info

**Returns:**
- `CompiledKernel`: A launchable kernel object

### `CompiledKernel`

A compiled kernel object.

**Methods:**
- `__call__(*args, stream=None)`: Launch the kernel
- `synchronize(stream=None)`: Synchronize the CUDA stream

## Notes

1. Ensure the CUDA toolkit and nvcc are installed on the system
2. Ensure a CUDA GPU is available
3. Supported argument types:
   - PyTorch tensors
   - NumPy arrays
   - Python int/float (automatically converted to the corresponding CUDA type)
4. Grid and block dimensions must be set correctly according to the kernel logic

## Examples

See `design/test_runtime.py` and `design/sm90_bf16_gemm.py`.
