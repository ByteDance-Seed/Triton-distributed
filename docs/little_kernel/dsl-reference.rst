DSL Reference
=============

.. contents:: On this page
   :local:
   :depth: 2

Kernel Definition
-----------------

Kernels are regular Python functions decorated with ``@lk.ll_kernel``:

.. code-block:: python

    import little_kernel as lk
    import little_kernel.language as ll

    @lk.ll_kernel(backend="cuda", is_entry=True)
    def my_kernel(
        data: ll.Tensor[ll.float32],
        N: ll.int32,
    ) -> ll.void:
        ...

Parameters:

- ``backend``: Currently only ``"cuda"`` is supported.
- ``is_entry``: ``True`` for top-level (launchable) kernels; ``False`` for
  device functions that can be called from other kernels.

Type System
-----------

Scalar Types
~~~~~~~~~~~~

=========== ================== =========
LK Type     C++ Type           Bytes
=========== ================== =========
``int32``   ``int``            4
``uint32``  ``unsigned int``   4
``int64``   ``long long``      8
``uint64``  ``unsigned long long`` 8
``float16`` ``__half``         2
``bfloat16`` ``__nv_bfloat16`` 2
``float32`` ``float``          4
``float64`` ``double``         8
``void``    ``void``           0
=========== ================== =========

Tensor Types
~~~~~~~~~~~~

``ll.Tensor[dtype]`` maps to a raw pointer (``dtype*``) in generated code.
All pointer arithmetic and bounds checking is the programmer's responsibility.

Qualifiers
~~~~~~~~~~

- ``ll.const[T]``: C++ ``const T``
- ``ll.grid_constant[T]``: ``__grid_constant__ const T``  (for TMA descriptors)

Variable Declarations
~~~~~~~~~~~~~~~~~~~~~

Use type annotations for local variables:

.. code-block:: python

    x: ll.float32 = 0.0
    idx: ll.int32 = ll.threadIdx_x()
    addr: ll.uint64 = ll.get_smem_address(buf)

Shared Memory
~~~~~~~~~~~~~

Allocate shared memory with ``ll.empty``:

.. code-block:: python

    # Static shared memory
    buf = ll.empty([1024], dtype=ll.float32, scope="shared")

    # Dynamic shared memory
    ll.align_memory(1024, scope="dynamic_shared")
    buf = ll.empty([SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

Control Flow
~~~~~~~~~~~~

Standard Python ``if/else``, ``for`` (with ``range()``), and ``while`` are
supported and map directly to C++ control flow.

Loop pragmas: ``ll.unroll()`` emits ``#pragma unroll``.

Building & Launching
--------------------

.. code-block:: python

    from little_kernel.core.passes import PASSES
    from little_kernel.codegen.codegen_cuda import codegen_cuda

    passes = PASSES["cuda"]
    kernel = my_kernel.build(
        passes, codegen_cuda,
        grid=(grid_x, grid_y, grid_z),
        block=(block_x, block_y, block_z),
        shared_mem_bytes=SMEM_SIZE,
        arch="sm_90",           # or "sm_100a" for Blackwell
        cluster_dim=(cx, cy, cz),  # optional, for SM90+ clusters
    )

    # Launch (pass same args as the @ll_kernel function)
    kernel(data_tensor, N)
    torch.cuda.synchronize()

Inspecting Generated Code
--------------------------

To see the generated CUDA source without compiling:

.. code-block:: python

    cuda_src = my_kernel.compile(passes, codegen_cuda)
    print(cuda_src)
