DSL Reference
=============

.. contents:: On this page
   :local:
   :depth: 2

Kernel Definition
-----------------

Kernels and device helpers are regular Python functions decorated with
``@lk.ll_kernel``:

.. code-block:: python

    import little_kernel as lk
    import little_kernel.language as ll

    @lk.ll_kernel(backend="nv/sm90", is_entry=True)
    def my_kernel(
        data: ll.Tensor[ll.float32],
        N: ll.int32,
    ) -> ll.void:
        ...

Decorator parameters:

- ``backend``: ``"nv/sm90"`` (**SM90**), ``"nv/sm100"`` (**SM100**), or
  ``"amd/cdna4"`` (**GFX950**); installed ``"cuda"`` / ``"hip"`` aliases remain
  supported.
- ``is_entry``: ``True`` for host-launchable ``__global__`` entry kernels;
  ``False`` for ``__device__`` helper functions callable from entry kernels.
- ``frontend``: Optional explicit frontend pass before ``PASSES[backend]``, such as
  ``dataflow_frontend`` on ``nv/sm100`` (see :doc:`dataflow`).

Type System
-----------

Scalar Types
~~~~~~~~~~~~

============ ====================== =========
LK Type      C++ Type               Bytes
============ ====================== =========
``int32``    ``int``                4
``uint32``   ``unsigned int``       4
``int64``    ``long long``          8
``uint64``   ``unsigned long long`` 8
``float16``  ``__half``             2
``bfloat16`` ``__nv_bfloat16``      2
``float32``  ``float``              4
``float64``  ``double``             8
``void``     ``void``               0
============ ====================== =========

Narrow integer, FP8/FP4 storage, and backend-specific vector types are also
available through each backend's language view (``get_backend(...).language``).

Tensor Types & Qualifiers
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``ll.Tensor[dtype]`` lowers to a typed raw pointer (``dtype*``). Pointer
  arithmetic and bounds checks remain explicit and predictable.
- ``ll.const[T]`` emits ``const T``.
- ``ll.grid_constant[T]`` emits ``__grid_constant__ const T`` (used for ``CUtensorMap``
  ``TMA`` descriptors on ``SM90`` and ``SM100``).

Shared Memory Allocation
------------------------

Allocate static or dynamic shared memory / LDS with ``ll.empty``:

.. code-block:: python

    # Static shared memory
    buf = ll.empty([1024], dtype=ll.float32, scope="shared")

    # Aligned dynamic shared memory
    ll.align_memory(1024, scope="dynamic_shared")
    tile = ll.empty([128, 64], dtype=ll.bfloat16, scope="dynamic_shared")

Control Flow & Loop-Carried Scalars
-----------------------------------

Because LittleKernel translates the Python AST rather than executing the function
for tracing, standard ``if/else``, ``while``, and ``for i in range(start, stop, step)``
with runtime bounds map directly to C++ control flow:

- Use ``ll.unroll(range(...))`` for compile-time loop unrolling or ``ll.unroll()``
  to emit ``#pragma unroll``.
- **Loop-carried mutable scalars**: Annotate a variable **once before the loop**
  and reassign it with plain ``=`` or ``+=`` inside the body (including inside
  nested ``if`` branches). Loop-local temporaries, by contrast, are annotated
  inside the loop body:

.. code-block:: python

    nxt: ll.int32 = ll.int32(src[tid])   # loop-carried prologue (annotated once)
    acc: ll.int32 = 0                    # loop-carried accumulator
    for i in range(tid, N, stride):
        cur: ll.int32 = nxt              # loop-local temporary (annotated in body)
        ahead: ll.int32 = i + stride
        if ahead < N:
            nxt = ll.int32(src[ahead])   # plain assignment mutates carried variable
            acc += 1
        dst[i] = cur + acc

Building, Launching & Source Inspection
---------------------------------------

.. code-block:: python

    from little_kernel.backends import get_backend
    from little_kernel.core.passes import PASSES

    backend = get_backend("nv/sm90")
    passes = PASSES[backend.backend_id]
    codegen_fn = backend.codegen.generate

    # Inspect generated CUDA/HIP C++ on CPU without compiling:
    src = my_kernel.compile(passes, codegen_fn)

    # Compile with NVCC/HIPCC and launch on GPU:
    kernel = my_kernel.build(
        passes,
        codegen_fn,
        grid=(grid_x, grid_y, grid_z),
        block=(block_x, block_y, block_z),
        shared_mem_bytes=smem_bytes,
        arch="sm_90a",                # "sm_100a" for SM100, "gfx950" for amd/cdna4
        cluster_shape=(cx, cy, cz),   # optional cluster launch for SM90 / SM100
    )
    kernel(data_tensor, N)
    kernel.synchronize()
