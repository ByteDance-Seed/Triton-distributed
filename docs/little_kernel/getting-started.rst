Getting Started
===============

Installation and CPU checks
--------------------------

The compiler SDK requires Python 3.11 or newer and NumPy. From the repository root::

    uv sync --locked --group cpu-test --group build --group format
    uv run --locked pytest -q
    uv run --locked pytest -q tests/repository

The root project also builds a standalone LittleKernel wheel. Triton-distributed
continues to use its existing ``python/setup.py`` entry point, with the bundled
compiler controlled by ``TRITON_BUILD_LITTLE_KERNEL``.

Writing a kernel
----------------

.. code-block:: python

    import little_kernel as lk
    from little_kernel.backends import get_backend
    from little_kernel.core.passes import PASSES

    backend = get_backend("nv/sm90")
    ll = backend.language

    @lk.ll_kernel(backend="nv/sm90", is_entry=True)
    def vector_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
                   out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
        i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
        if i < n:
            out[i] = a[i] + b[i]

    artifact = vector_add.compile_artifact(
        PASSES[backend.backend_id], backend.codegen.generate,
        target="sm_90a", num_threads=256,
    )
    print(artifact.source)

Source generation does not require a device or tensor framework. ``build``
additionally invokes the selected backend toolchain and returns a callable
executable. Use ``nv/sm100`` for Blackwell or ``amd/cdna4`` for gfx950.

Hardware checks and benchmarks
------------------------------

Declare the matching tensor/benchmark integration in the intended environment's
dependency configuration and lock. NVCC/HIPCC, the driver and the physical device
must match the selected hardware. The CPU lock does not provision that stack::

    uv run --locked python tools/verify.py --backend nv/sm90 --suite gpu
    uv run --locked python tools/verify.py --backend amd/cdna4 --suite gpu
    uv run --locked python -m benchmarks.nv.common.run_all --list
    uv run --locked python -m benchmarks.nv.sm90.gemm.sm90_manual_wgmma_swizzle
    uv run --locked python -m benchmarks.nv.sm100.gemm.run_sm100_levels

Missing hardware is unavailable, never a successful numerical/performance check.
