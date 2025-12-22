AllReduce
=========

Distributed AllReduce kernel implementations for NVIDIA GPUs.

API Reference
-------------

Multiple AllReduce methods are supported:

- **one_shot**: Single-pass AllReduce
- **two_shot**: Two-pass AllReduce for larger data
- **double_tree**: Double-tree algorithm for balanced communication
- **one_shot_tma**: TMA-based single-pass AllReduce
- **one_shot_multimem**: Multicast memory-based single-pass AllReduce  
- **two_shot_multimem**: Multicast memory-based two-pass AllReduce

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.nvidia.allreduce import allreduce_kernel

   # Run AllReduce test
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot

