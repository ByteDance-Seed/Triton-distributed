triton-dist.language
====================

This page documents all Python APIs in Triton-distributed.

triton_dist.language
--------------------

Core distributed language extensions for Triton.

Distributed Operations
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: wait(barrierPtrs, numBarriers, scope, semantic, waitValue=1)

   Wait for distributed barriers.

   :param barrierPtrs: Pointer to barrier signals
   :param numBarriers: Number of barriers to wait for
   :param scope: Memory scope - ``"gpu"`` or ``"sys"``
   :param semantic: Memory semantic - ``"acquire"``, ``"release"``, etc.
   :param waitValue: Value to wait for (default: 1)
   :returns: Token for consume_token

.. py:function:: consume_token(value, token)

   Consume a token to ensure proper ordering with wait operations.

   :param value: Value (pointer or tensor) to consume
   :param token: Token from wait operation
   :returns: The consumed value

.. py:function:: rank(axis=-1)

   Get the current rank ID.

   :param axis: Axis for multi-dimensional rank (default: -1 for global rank)
   :returns: Rank ID as int32

.. py:function:: num_ranks(axis=-1)

   Get the total number of ranks.

   :param axis: Axis for multi-dimensional rank count
   :returns: Number of ranks as int32

.. py:function:: symm_at(ptr, rank)

   Map a symmetric pointer to a remote rank.

   :param ptr: Local symmetric pointer
   :param rank: Target rank ID
   :returns: Pointer on the target rank

.. py:function:: notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node")

   Send a signal to a remote rank.

   :param ptr: Signal pointer
   :param rank: Target rank ID
   :param signal: Signal value (default: 1)
   :param sig_op: Signal operation - ``"set"`` or ``"add"``
   :param comm_scope: Communication scope - ``"intra_node"`` or ``"inter_node"``

SIMT Operations
^^^^^^^^^^^^^^^

.. py:function:: simt_exec_region()

   Context manager for SIMT execution region.

   Usage::

       with simt_exec_region():
           # SIMT code here
           pass

.. py:class:: vector

   Vector type for SIMT operations.

   .. py:method:: __init__(args)

      Create a vector from a sequence of tensors.

   .. py:method:: __getitem__(idx)

      Get element at index.

   .. py:method:: __setitem__(idx, value)

      Set element at index.

   .. py:method:: __add__(other)

      Element-wise addition.

   .. py:method:: __sub__(other)

      Element-wise subtraction.

   .. py:method:: __mul__(other)

      Element-wise multiplication.

   .. py:method:: recast(new_elem_dtype)

      Bitcast to a different element type.

   .. py:method:: to(dtype, fp_downcast_rounding=None, bitcast=False)

      Convert to a different dtype.

.. py:function:: make_vector(args)

   Create a vector from arguments.

   :param args: Sequence of tensor values
   :returns: vector object

.. py:function:: zeros_vector(vec_size, dtype)

   Create a zero-initialized vector.

   :param vec_size: Vector size (constexpr)
   :param dtype: Element data type
   :returns: vector of zeros

.. py:function:: extract(input, indices)

   Extract a scalar from a tile at given indices.

   :param input: Input tensor
   :param indices: List of indices
   :returns: Scalar value

.. py:function:: insert(input, scalar, indices)

   Insert a scalar into a tile at given indices.

   :param input: Input tensor
   :param scalar: Scalar value to insert
   :param indices: List of indices
   :returns: Modified tensor

Core Operations
^^^^^^^^^^^^^^^

.. py:function:: extern_call(lib_name, lib_path, args, arg_type_symbol_dict, is_pure)

   Call an external library function.

   :param lib_name: Name of the library
   :param lib_path: Path to the library
   :param args: Function arguments
   :param arg_type_symbol_dict: Type-to-symbol mapping
   :param is_pure: Whether the function is pure (no side effects)

triton_dist.language.extra
--------------------------

Extra language utilities.

language_extra Module
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: tid(axis)

   Get the thread index within a CTA.

   :param axis: Axis (0 for threadIdx.x)
   :returns: Thread index

.. py:function:: atomic_cas(ptr, cmp_val, target_val, scope, semantic)

   Atomic compare-and-swap.

   :param ptr: Target pointer
   :param cmp_val: Compare value
   :param target_val: Target value if comparison succeeds
   :param scope: Memory scope
   :param semantic: Memory semantic

.. py:function:: atomic_add(ptr, val, scope, semantic)

   Atomic add operation.

   :param ptr: Target pointer
   :param val: Value to add
   :param scope: Memory scope
   :param semantic: Memory semantic

.. py:function:: __syncthreads()

   Block-level synchronization barrier.

.. py:function:: ld(ptr, scope, semantic)

   Load with memory ordering.

   :param ptr: Pointer to load from
   :param scope: Memory scope
   :param semantic: Memory semantic

.. py:function:: st(ptr, val, scope, semantic)

   Store with memory ordering.

   :param ptr: Pointer to store to
   :param val: Value to store
   :param scope: Memory scope
   :param semantic: Memory semantic

.. py:function:: ld_vector(ptr, vec_size, scope, semantic)

   Load a vector.

   :param ptr: Pointer to load from
   :param vec_size: Vector size (constexpr)
   :param scope: Memory scope
   :param semantic: Memory semantic
   :returns: vector object

.. py:function:: st_vector(ptr, vec, scope, semantic)

   Store a vector.

   :param ptr: Pointer to store to
   :param vec: vector object to store
   :param scope: Memory scope
   :param semantic: Memory semantic

.. py:function:: pack(src, dst_type)

   Pack values into a larger type.

.. py:function:: unpack(src, dst_type)

   Unpack a value into smaller types.

.. py:function:: threads_per_warp()

   Get threads per warp (32 for NVIDIA, 64 for AMD).

.. py:function:: num_threads()

   Get total number of threads (num_warps * threads_per_warp).

.. py:function:: num_warps()

   Get number of warps.

libshmem_device Module
^^^^^^^^^^^^^^^^^^^^^^

NVSHMEM/ROCSHMEM device API bindings. The API automatically dispatches to the appropriate backend.

**Initialization:**

.. py:function:: my_pe()

   Get current PE (processing element) ID.

.. py:function:: n_pes()

   Get total number of PEs.

.. py:function:: team_my_pe(team)

   Get PE ID within a team.

.. py:function:: team_n_pes(team)

   Get number of PEs in a team.

**Memory Operations:**

.. py:function:: remote_ptr(local_ptr, pe)

   Get remote pointer on another PE.

.. py:function:: remote_mc_ptr(team, ptr)

   Get multicast remote pointer.

**Synchronization:**

.. py:function:: barrier_all()

   Global barrier across all PEs.

.. py:function:: barrier_all_block()

   Block-level global barrier.

.. py:function:: barrier_all_warp()

   Warp-level global barrier.

.. py:function:: barrier(team)

   Barrier within a team.

.. py:function:: sync_all()

   Global sync across all PEs.

.. py:function:: quiet()

   Ensure all outstanding operations complete.

.. py:function:: fence()

   Memory fence.

**Get Operations:**

.. py:function:: getmem_nbi_block(dest, source, bytes, pe)

   Non-blocking block-level get.

.. py:function:: getmem_block(dest, source, bytes, pe)

   Blocking block-level get.

.. py:function:: getmem_nbi_warp(dest, source, bytes, pe)

   Non-blocking warp-level get.

.. py:function:: getmem_warp(dest, source, bytes, pe)

   Blocking warp-level get.

.. py:function:: getmem_nbi(dest, source, bytes, pe)

   Non-blocking get.

.. py:function:: getmem(dest, source, bytes, pe)

   Blocking get.

**Put Operations:**

.. py:function:: putmem_block(dest, source, bytes, pe)

   Block-level put.

.. py:function:: putmem_nbi_block(dest, source, bytes, pe)

   Non-blocking block-level put.

.. py:function:: putmem_warp(dest, source, bytes, pe)

   Warp-level put.

.. py:function:: putmem_nbi_warp(dest, source, bytes, pe)

   Non-blocking warp-level put.

.. py:function:: putmem(dest, source, bytes, pe)

   Put operation.

.. py:function:: putmem_nbi(dest, source, bytes, pe)

   Non-blocking put.

**Signal Operations:**

.. py:function:: putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe)

   Non-blocking block-level put with signal.

.. py:function:: putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe)

   Block-level put with signal.

.. py:function:: signal_op(sig_addr, signal, sig_op, pe)

   Signal operation.

.. py:function:: signal_wait_until(sig_addr, cmp_, cmp_val)

   Wait until signal meets condition.

**Collective Operations:**

.. py:function:: broadcastmem_block(team, dest, source, nelems, pe_root)

   Block-level broadcast.

.. py:function:: fcollectmem_block(team, dest, source, nelems)

   Block-level fixed collect.

**Constants:**

.. py:data:: NVSHMEM_CMP_EQ

   Compare equal

.. py:data:: NVSHMEM_CMP_NE

   Compare not equal

.. py:data:: NVSHMEM_CMP_GT

   Compare greater than

.. py:data:: NVSHMEM_CMP_LT

   Compare less than

.. py:data:: NVSHMEM_CMP_GE

   Compare greater or equal

.. py:data:: NVSHMEM_CMP_LE

   Compare less or equal

.. py:data:: NVSHMEM_SIGNAL_SET

   Signal set operation

.. py:data:: NVSHMEM_SIGNAL_ADD

   Signal add operation

.. py:data:: NVSHMEM_TEAM_WORLD

   World team

.. py:data:: NVSHMEMX_TEAM_NODE

   Node team

triton_dist.utils
-----------------

Utility functions for distributed programming.

Initialization
^^^^^^^^^^^^^^

.. py:function:: initialize_distributed(seed=None, initialize_shmem=True)

   Initialize distributed runtime.

   :param seed: Random seed (default: rank)
   :param initialize_shmem: Whether to initialize NVSHMEM/ROCSHMEM
   :returns: Tensor parallel process group

.. py:function:: init_nvshmem_by_torch_process_group(pg)

   Initialize NVSHMEM using a PyTorch process group.

.. py:function:: init_rocshmem_by_torch_process_group(pg)

   Initialize ROCSHMEM using a PyTorch process group.

.. py:function:: finalize_distributed()

   Finalize distributed runtime.

.. py:function:: is_shmem_initialized()

   Check if SHMEM is initialized.

Tensor Allocation
^^^^^^^^^^^^^^^^^

.. py:function:: nvshmem_create_tensor(shape, dtype)

   Create a symmetric tensor with NVSHMEM.

   :param shape: Tensor shape
   :param dtype: Data type
   :returns: Symmetric torch.Tensor

.. py:function:: nvshmem_create_tensors(shape, dtype, rank, local_world_size)

   Create a list of peer tensors.

   :returns: List of tensors for each local rank

.. py:function:: nvshmem_free_tensor_sync(tensor)

   Free a symmetric tensor.

Synchronization
^^^^^^^^^^^^^^^

.. py:function:: nvshmem_barrier_all_on_stream(stream=None)

   Barrier all on a CUDA stream.

.. py:function:: rocshmem_barrier_all_on_stream(stream=None)

   Barrier all on a HIP stream.

Platform Detection
^^^^^^^^^^^^^^^^^^

.. py:function:: is_cuda()

   Check if running on CUDA platform.

.. py:function:: is_hip()

   Check if running on HIP platform.

.. py:function:: has_tma()

   Check if TMA is supported (SM90+).

.. py:function:: is_nvshmem_multimem_supported()

   Check if NVSHMEM multicast memory is supported.

Device Info
^^^^^^^^^^^

.. py:function:: get_numa_node(device_id)

   Get NUMA node for a device.

.. py:function:: get_max_gpu_clock_rate_in_khz(device_id)

   Get maximum GPU clock rate.

.. py:function:: get_current_gpu_clock_rate_in_khz(device_id)

   Get current GPU clock rate.

.. py:function:: get_device_max_shared_memory_size(device_id)

   Get maximum shared memory size.

.. py:function:: has_fullmesh_nvlink()

   Check if NVLink full mesh is available.

.. py:function:: supports_p2p_native_atomic()

   Check if P2P native atomic is supported.

Utility Functions
^^^^^^^^^^^^^^^^^

.. py:function:: dist_print(*args, allowed_ranks=[0], prefix=False, need_sync=False, **kwargs)

   Print from specific ranks.

   :param allowed_ranks: List of ranks to print from, or "all"
   :param prefix: Whether to prefix with rank
   :param need_sync: Whether to synchronize before printing

.. py:function:: rand_tensor(shape, dtype, device="cuda")

   Generate random tensor.

   :param shape: Tensor shape
   :param dtype: Data type
   :param device: Device
   :returns: Random tensor with values in [-1, 1] for floats

.. py:function:: init_seed(seed=0)

   Initialize random seeds for reproducibility.

.. py:function:: get_bool_env(env, default_value)

   Get boolean environment variable.

.. py:function:: get_int_env(env, default_value)

   Get integer environment variable.

.. py:data:: NVSHMEM_SIGNAL_DTYPE

   Default signal dtype (torch.int64)

triton_dist.tools
-----------------

Development and debugging tools.

Compile Tools
^^^^^^^^^^^^^

.. py:function:: make_ast_source(...)

   Generate AST source for AOT compilation.

.. py:function:: kernel_name_suffix(...)

   Generate kernel name suffix.

.. py:function:: materialize_c_params(...)

   Materialize C parameters.

.. py:function:: dump_c_code(...)

   Dump generated C code.

.. py:function:: aot_compile_spaces(...)

   AOT compile with configuration spaces.

.. py:function:: apply_triton340_inductor_patch()

   Apply compatibility patch for torch.compile with Triton 3.4.0.

Profiler
^^^^^^^^

.. py:class:: Profiler

   Intra-kernel profiler for performance analysis.

.. py:function:: alloc_profiler_buffer(...)

   Allocate profiler buffer.

.. py:function:: reset_profiler_buffer(...)

   Reset profiler buffer.

.. py:class:: ProfilerBuffer

   Profiler buffer class.

.. py:function:: export_to_perfetto_trace(...)

   Export profile to Perfetto trace format.

.. py:function:: parse_to_tracks(...)

   Parse profile data to tracks.

triton_dist.mega_triton_kernel
------------------------------

MegaTritonKernel for fused multi-layer kernels.

Core
^^^^

.. py:class:: ModelBuilder

   Build fused models with MegaTritonKernel.

Tasks
^^^^^

Available tasks for MegaTritonKernel:

- Activation tasks
- AllReduce tasks  
- Barrier tasks
- Elementwise tasks
- Flash Attention tasks
- Flash Decode tasks
- Linear tasks
- Normalization tasks
- Prefetch tasks

See ``python/triton_dist/mega_triton_kernel/tasks/`` for implementation details.

Example Usage
-------------

.. code-block:: python

   import triton_dist
   import triton.language as tl
   import triton_dist.language as dl
   from triton_dist.language.extra import libshmem_device
   from triton_dist.language.extra.language_extra import tid, __syncthreads

   @triton_dist.jit
   def my_distributed_kernel(input_ptr, output_ptr, ...):
       rank = dl.rank()
       num_ranks = dl.num_ranks()
       
       # Use symmetric pointers
       remote_ptr = dl.symm_at(input_ptr, peer_rank)
       
       # Wait for data
       token = dl.wait(signal_ptr, 1, "sys", "acquire")
       input_ptr = dl.consume_token(input_ptr, token)
       
       # Notify completion
       dl.notify(signal_ptr, peer_rank, signal=1, sig_op="set", comm_scope="intra_node")
