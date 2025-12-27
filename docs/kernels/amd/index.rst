AMD Kernels
===========

This section documents all available kernels for AMD GPUs (CDNA3).

Kernel List
-----------

.. list-table:: Available AMD Kernels
   :header-rows: 1
   :widths: 30 70

   * - Kernel
     - Description
   * - :doc:`all_gather_gemm`
     - AllGather + GEMM with computation-communication overlapping (intra-node)
   * - :doc:`gemm_reduce_scatter`
     - GEMM + ReduceScatter with computation-communication overlapping (intra-node)
   * - :doc:`gemm_allreduce`
     - GEMM + AllReduce kernel

.. toctree::
   :maxdepth: 1
   :hidden:

   all_gather_gemm
   gemm_reduce_scatter
   gemm_allreduce

