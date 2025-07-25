.. _sphx_glr_getting-started_tutorials_10-AMD-overlapping-gemm-reduce-scatter.rst:

Overlapping GEMM ReduceScatter on AMD GPU
=========================================

In this tutorial, you will write a fused Gemm and ReduceScatter Op using Triton-distributed.

In doing so, you will learn about:

* How to overlap reduce-scatter with gemm operations to hide communication on AMD GPUs.

.. code-block:: bash

    # To run this tutorial
    bash ./scripts/launch_amd.sh ./tutorials/10-AMD-overlapping-gemm-reduce-scatter.py

