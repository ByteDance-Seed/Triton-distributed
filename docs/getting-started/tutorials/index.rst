:orphan:

Tutorials
=========

We provide a list tutorials for writing various distributed operations with Triton-distributed.
It is recommended that you first read the technique report, which contains design and implementation details, and then play with these tutorials.


1. [Primitives]: `Basic notify and wait operation <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/01-distributed-notify-wait.py>`_

2. [Primitives & Communication]: `Use copy engine and NVSHMEM primitives for AllGather <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/02-intra-node-allgather.py>`_

3. [Communication]: `Inter-node AllGather <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/03-inter-node-allgather.py>`_

4. [Communication]: `Intra-node and Inter-node DeepSeek EP AllToAll <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/04-deepseek-infer-all2all.py>`_

5. [Communication]: `Intra-node ReduceScatter <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/05-intra-node-reduce-scatter.py>`_

6. [Communication]: `Inter-node ReduceScatter <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/06-inter-node-reduce-scatter.py>`_

7. [Overlapping]: `AllGather GEMM overlapping <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/07-overlapping-allgather-gemm.py>`_

8. [Overlapping]: `GEMM ReduceScatter overlapping <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/08-overlapping-gemm-reduce-scatter.py>`_

9. [Overlapping]: `AllGather GEMM overlapping on AMD <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/09-AMD-overlapping-allgather-gemm.py>`_

10. [Overlapping]: `GEMM ReduceScatter overlapping on AMD <https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/10-AMD-overlapping-gemm-reduce-scatter.py>`_

.. toctree::
   :hidden:

   /getting-started/tutorials/01-distributed-notify-wait
   /getting-started/tutorials/02-intra-node-allgather