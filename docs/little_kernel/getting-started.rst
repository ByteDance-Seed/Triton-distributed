Getting Started
===============

.. contents:: On this page
   :local:
   :depth: 2

Installation
------------

LittleKernel is included in Triton-distributed.  A standard editable install
picks it up automatically::

    cd python
    TRITON_BUILD_LITTLE_KERNEL=ON pip install -e .

Verify::

    python -c "import little_kernel; print('OK')"

Prerequisites:

- Python >= 3.9
- PyTorch with CUDA support
- ``nvcc`` on ``$PATH`` (CUDA Toolkit 12+; CUDA 13 for SM100)

Writing Your First Kernel
-------------------------

A minimal kernel that adds two vectors:

.. code-block:: python

    import little_kernel as lk
    import little_kernel.language as ll
    from little_kernel.core.passes import PASSES
    from little_kernel.codegen.codegen_cuda import codegen_cuda
    import torch

    @lk.ll_kernel(backend="cuda", is_entry=True)
    def vec_add(
        A: ll.Tensor[ll.float32],
        B: ll.Tensor[ll.float32],
        C: ll.Tensor[ll.float32],
        N: ll.int32,
    ) -> ll.void:
        tid: ll.int32 = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
        if tid < N:
            C[tid] = A[tid] + B[tid]

    # Build
    passes = PASSES["cuda"]
    N = 1024
    kernel = vec_add.build(
        passes, codegen_cuda,
        grid=(N // 256, 1, 1),
        block=(256, 1, 1),
        arch="sm_90",
    )

    # Launch
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty(N, device="cuda", dtype=torch.float32)
    kernel(A, B, C, N)
    torch.cuda.synchronize()
    assert torch.allclose(C, A + B)
    print("PASS")

Running Benchmarks
------------------

Micro-benchmarks
~~~~~~~~~~~~~~~~

Run the full suite on a Hopper GPU::

    python -m little_kernel.benchmark.run_all

Or a specific category::

    python -m little_kernel.benchmark.run_all --category compute

SM90 GEMM Kernels
~~~~~~~~~~~~~~~~~

Run individual GEMM variants::

    python -m little_kernel.benchmark.gemm_sm90.gemm_v1

SM100 GEMM Kernels
~~~~~~~~~~~~~~~~~~

Run all levels (requires Blackwell GPU)::

    python -m little_kernel.benchmark.gemm_sm100.test_all_levels

Or a specific level::

    python -m little_kernel.benchmark.gemm_sm100.gemm_level1

Running Tests
-------------

Unit tests (no GPU required for most)::

    pytest test/little_kernel/unit/ -v

Integration tests (requires GPU)::

    pytest test/little_kernel/integration/ -v
