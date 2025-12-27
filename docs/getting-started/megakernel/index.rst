:orphan:

Megakernel Implementations
==========================

Here you'll find a list of tutorials for implementing MegaKernels in Triton-distributed.

1. :doc:`Megakernel <megakernel>`: We provide a tutorial of MegaKernel for a dense language model (e.g., Qwen3-32B) by integrating our tensor parallelism modules.

2. **EP MoE Megakernel**: Expert Parallelism MoE megakernel implementations for single-node 8-GPU configurations:
   
   - :doc:`EP All-to-All Fused Kernel </kernels/nvidia/ep_all2all_fused>`: Fused megakernel combining dispatch+groupgemm and groupgemm+combine operations with token optimization (token saving/skipping, token sorting, SM scheduling)
   
   - :doc:`EP All-to-All Fused Layer </layers/nvidia/ep_a2a_fused_layer>`: High-level layer API for fused EP MoE operations


.. toctree::
   :hidden:

   /getting-started/megakernel/megakernel