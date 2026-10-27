AllGather
=========

Low-latency AllGather kernel implementations for NVIDIA GPUs.

API Reference
-------------

.. py:function:: fast_allgather(symm_buffer, ctx=None, rank=None, node=None, num_ranks=None, num_nodes=None, mode="pull")

   Performs fast AllGather using the selected mode. Pass ``ctx`` or provide
   ``rank``, ``node``, ``num_ranks``, and ``num_nodes`` to create a context.

   :param symm_buffer: Symmetric buffer to AllGather
   :param ctx: Optional FastAllGatherContext
   :param rank: Current rank, required when ``ctx`` is not provided
   :param node: Current node, required when ``ctx`` is not provided
   :param num_ranks: Total number of ranks, required when ``ctx`` is not provided
   :param num_nodes: Total number of nodes, required when ``ctx`` is not provided
   :param mode: AllGather implementation (default: ``"pull"``)

.. py:function:: create_fast_allgather_context(...)

   Creates the context for fast AllGather.

.. py:function:: get_auto_all_gather_method(...)

   Automatically selects the best AllGather method based on hardware topology.

.. py:class:: AllGatherMethod

   Enum for AllGather methods:
   
   - ``PULL``: Pull-based AllGather
   - ``PUSH_2D``: 2D push-based AllGather
   - ``PUSH_3D``: 3D push-based AllGather
   - ``PUSH_2D_LL``: Low-latency 2D push
   - ``PUSH_2D_LL_MULTIMEM``: Low-latency 2D push with multicast memory
   - ``PUSH_NUMA_2D``: NUMA-aware 2D push
   - ``PUSH_NUMA_2D_LL``: NUMA-aware low-latency 2D push

Internal Kernels
----------------

.. py:function:: _forward_pull_kernel(...)

   Pull-based AllGather kernel.

.. py:function:: _forward_push_2d_kernel(...)

   2D push-based AllGather kernel.

.. py:function:: _forward_push_3d_kernel(...)

   3D push-based AllGather kernel.

.. py:function:: _forward_push_2d_ll_kernel(...)

   Low-latency 2D push AllGather kernel.

.. py:function:: _forward_push_2d_ll_multimem_kernel(...)

   Low-latency 2D push AllGather with multicast memory.

.. py:function:: _forward_push_numa_2d_kernel(...)

   NUMA-aware 2D push AllGather kernel.

.. py:function:: _forward_push_numa_2d_ll_kernel(...)

   NUMA-aware low-latency 2D push AllGather kernel.

.. py:function:: cp_engine_producer_all_gather_intra_node(...)

   Copy engine-based intra-node AllGather.

.. py:function:: cp_engine_producer_all_gather_inter_node(...)

   Copy engine-based inter-node AllGather.

