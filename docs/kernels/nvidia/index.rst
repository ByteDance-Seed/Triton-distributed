NVIDIA Kernels
==============

This section documents all available kernels for NVIDIA GPUs (SM80, SM89, SM90a).

Kernel List
-----------

.. list-table:: Available NVIDIA Kernels
   :header-rows: 1
   :widths: 30 70

   * - Kernel
     - Description
   * - :doc:`allgather_gemm`
     - AllGather + GEMM with computation-communication overlapping
   * - :doc:`gemm_reduce_scatter`
     - GEMM + ReduceScatter with computation-communication overlapping
   * - :doc:`allgather`
     - Low-latency AllGather implementations (pull, push 2D/3D, multimem)
   * - :doc:`allreduce`
     - Distributed AllReduce kernels (one-shot, two-shot, double-tree, multimem)
   * - :doc:`flash_decode`
     - Distributed Flash Decoding for attention
   * - :doc:`ep_a2a`
     - Expert Parallelism All-to-All for MoE models (inter-node)
   * - :doc:`ep_a2a_intra_node`
     - Expert Parallelism All-to-All (intra-node optimized)
   * - :doc:`low_latency_a2a_v2`
     - Low-Latency EP All-to-All with FP8 quantization
   * - :doc:`gemm_allreduce`
     - GEMM + AllReduce with computation-communication overlapping
   * - :doc:`moe_reduce_rs`
     - MoE ReduceScatter kernel
   * - :doc:`moe_reduce_ar`
     - MoE AllReduce kernel
   * - :doc:`sp_ag_attention`
     - Sequence Parallel AllGather Attention (intra-node and inter-node)
   * - :doc:`ulysses_sp`
     - Ulysses-style Sequence Parallelism kernels

.. toctree::
   :maxdepth: 1
   :hidden:

   allgather_gemm
   gemm_reduce_scatter
   allgather
   allreduce
   flash_decode
   ep_a2a
   ep_a2a_intra_node
   low_latency_a2a_v2
   gemm_allreduce
   moe_reduce_rs
   moe_reduce_ar
   sp_ag_attention
   ulysses_sp

