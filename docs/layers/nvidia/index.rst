NVIDIA Layers
=============

High-level layer abstractions for NVIDIA GPUs.

Layer List
----------

.. list-table:: Available NVIDIA Layers
   :header-rows: 1
   :widths: 30 70

   * - Layer
     - Description
   * - :doc:`tp_attn`
     - Tensor Parallel Attention layer
   * - :doc:`tp_mlp`
     - Tensor Parallel MLP layer
   * - :doc:`tp_moe`
     - Tensor Parallel MoE layer
   * - :doc:`sp_flash_decode`
     - Sequence Parallel Flash Decode layer
   * - :doc:`ep_a2a_layer`
     - Expert Parallelism All-to-All layer
   * - :doc:`ep_a2a_fused_layer`
     - EP All-to-All fused layer (megakernel with token optimization)
   * - :doc:`ep_ll_a2a_layer`
     - Low-Latency Expert Parallelism All-to-All layer
   * - :doc:`gemm_allreduce_layer`
     - GEMM + AllReduce layer
   * - :doc:`low_latency_allgather_layer`
     - Low-Latency AllGather layer
   * - :doc:`ulysses_sp_a2a_layer`
     - Ulysses SP All-to-All layer
   * - :doc:`pp_block`
     - Pipeline Parallel block

.. toctree::
   :maxdepth: 1
   :hidden:

   tp_attn
   tp_mlp
   tp_moe
   sp_flash_decode
   ep_a2a_layer
   ep_a2a_fused_layer
   ep_ll_a2a_layer
   gemm_allreduce_layer
   low_latency_allgather_layer
   ulysses_sp_a2a_layer
   pp_block

