Getting Started
===============

.. contents:: On this page
   :local:
   :depth: 2

Installation & CPU Verification
-------------------------------

LittleKernel requires Python 3.11 or newer and NumPy for CPU compilation and
source generation. When building ``Triton-distributed`` via ``python/setup.py``,
the bundled ``little_kernel`` package (including ``nv/sm90``, ``nv/sm100``, and
``amd/cdna4`` descriptors and runtime headers) is installed automatically unless
disabled via ``TRITON_BUILD_LITTLE_KERNEL=0``.

To set up the standalone locked developer environment and run CPU verification
from the repository root:

.. code-block:: bash

    uv sync --locked --group cpu-test --group build --group format
    uv run --locked pytest -q
    uv run --locked python tools/verify.py --list
    uv run --locked python tools/verify.py --backend nv/sm90 --suite cpu
    uv run --locked python tools/verify.py --backend nv/sm100 --suite cpu
    uv run --locked python tools/verify.py --backend amd/cdna4 --suite cpu

Writing & Inspecting Your First Kernel
--------------------------------------

Source generation (``compile_artifact`` or ``compile``) runs purely on CPU without
requiring a GPU or PyTorch:

.. code-block:: python

    import little_kernel as lk
    from little_kernel.backends import get_backend
    from little_kernel.core.passes import PASSES

    backend = get_backend("nv/sm90")  # or "nv/sm100" / "amd/cdna4"
    ll = backend.language

    @lk.ll_kernel(backend="nv/sm90", is_entry=True)
    def vector_add(
        a: ll.Tensor[ll.float32],
        b: ll.Tensor[ll.float32],
        out: ll.Tensor[ll.float32],
        n: ll.int32,
    ) -> ll.void:
        i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
        if i < n:
            out[i] = a[i] + b[i]

    artifact = vector_add.compile_artifact(
        PASSES[backend.backend_id],
        backend.codegen.generate,
        target="sm_90a",
        num_threads=256,
    )
    print(artifact.source)

Building & Launching on GPU
---------------------------

Calling ``.build(...)`` invokes the backend's native compiler toolchain (``nvcc``
for ``SM90`` / ``SM100``, ``hipcc`` for ``GFX950``), caches the compiled binary,
and returns a callable launcher accepting PyTorch CUDA/HIP tensors and Python
scalars:

.. code-block:: python

    import torch

    n = 4096
    kernel = vector_add.build(
        PASSES["nv/sm90"],
        backend.codegen.generate,
        grid=((n + 255) // 256, 1, 1),
        block=(256, 1, 1),
        arch="sm_90a",  # use "sm_100a" on SM100 or "gfx950" on amd/cdna4
    )

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)
    kernel(a, b, out, n)
    kernel.synchronize()
    torch.testing.assert_close(out, a + b)

Running Hardware Suites & Benchmarks
------------------------------------

Device verification and benchmarks require matching physical hardware (**SM90**,
**SM100**, or **GFX950**), drivers, ``nvcc``/``hipcc``, and PyTorch:

.. code-block:: bash

    # Device verification suites
    uv run --locked python tools/verify.py --backend nv/sm90 --suite gpu
    uv run --locked python tools/verify.py --backend nv/sm100 --suite gpu
    uv run --locked python tools/verify.py --backend amd/cdna4 --suite gpu

    # Representative SM90 benchmarks
    python3 benchmarks/nv/sm90/gemm/sm90_deepgemm_full_optimizations.py
    python3 benchmarks/nv/sm90/flashattention/level6/flash_attn_varlen.py

    # Representative SM100 single-GPU & multi-GPU overlap benchmarks
    python3 benchmarks/nv/sm100/dataflow/bench_gemm_sm100.py
    python3 benchmarks/nv/sm100/dataflow/bench_mxfp8_gemm.py
    torchrun --standalone --nproc_per_node=4 benchmarks/nv/sm100/dataflow/bench_all_gather_gemm_sm100.py --no-triton-dist --no-thunderkittens

    # Representative GFX950 benchmarks
    python3 benchmarks/amd/cdna4/mxfp8_gemm_gfx950_persistent.py
