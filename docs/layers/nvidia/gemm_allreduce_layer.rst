GEMM AllReduce Layer
====================

High-level layer for GEMM + AllReduce fusion.

Description
-----------

This layer fuses GEMM computation with AllReduce communication for efficient
tensor parallelism in dense models.

See ``python/triton_dist/layers/nvidia/gemm_allreduce_layer.py`` for implementation details.

