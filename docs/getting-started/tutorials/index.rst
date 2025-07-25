:orphan:

Tutorials
=========

We provide a list tutorials for writing various distributed operations with Triton-distributed.
It is recommended that you first read the technique report, which contains design and implementation details, and then play with these tutorials.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a simple notify and wait example using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_01-distributed-notify-wait_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_01-distributed-notify-wait.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Distributed Notify and Wait</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a distributed AllGather kernel using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_02-intra-node-allgather_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_02-intra-node-allgather.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intra-node AllGather</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a low latency all gather kernel using using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_03-inter-node-allgather_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_03-inter-node-allgather.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Inter-node AllGather</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we demonstrate how to implement the All-to-All communication paradigm in Expert Parallelism (EP) for MoE models using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_04-deepseek-infer-all2all_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_04-deepseek-infer-all2all.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Low Latency All-to-All Communication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a intra-node reduce-scatter operation.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_05-intra-node-reduce-scatter_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_05-intra-node-reduce-scatter.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intra-node ReduceScatter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a multi-node reduce-scatter operation.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_06-inter-node-reduce-scatter_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_06-inter-node-reduce-scatter.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Inter-node ReduceScatter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a simple Allgather GEMM fusion kernel using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_07-overlapping-allgather-gemm_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_07-overlapping-allgather-gemm.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Overlapping AllGather GEMM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a Multi-node Gemm reduce-scatter operation that is significantly faster than PyTorch's native op.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_08-overlapping-gemm-reduce-scatter_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_08-overlapping-gemm-reduce-scatter.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Overlapping GEMM ReduceScatter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a simple fused AllGather and Gemm using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_09-AMD-overlapping-allgather-gemm_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_09-AMD-overlapping-allgather-gemm.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Overlapping AllGather GEMM on AMD GPU</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, you will write a fused Gemm and ReduceScatter Op using Triton-distributed.">

.. only:: html

  .. image:: /getting-started/tutorials/images/thumb/sphx_glr_10-AMD-overlapping-gemm-reduce-scatter_thumb.png
    :alt:

  :ref:`sphx_glr_getting-started_tutorials_10-AMD-overlapping-gemm-reduce-scatter.rst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Overlapping GEMM ReduceScatter on AMD GPU</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /getting-started/tutorials/01-distributed-notify-wait
   /getting-started/tutorials/02-intra-node-allgather
   /getting-started/tutorials/03-inter-node-allgather
   /getting-started/tutorials/04-deepseek-infer-all2all
   /getting-started/tutorials/05-intra-node-reduce-scatter
   /getting-started/tutorials/06-inter-node-reduce-scatter
   /getting-started/tutorials/07-overlapping-allgather-gemm
   /getting-started/tutorials/08-overlapping-gemm-reduce-scatter
   /getting-started/tutorials/09-AMD-overlapping-allgather-gemm
   /getting-started/tutorials/10-AMD-overlapping-gemm-reduce-scatter


.. 1. [Primitives]: `Basic notify and wait operation <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/01-distributed-notify-wait.py>`_

.. 2. [Primitives & Communication]: `Use copy engine and NVSHMEM primitives for AllGather <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/02-intra-node-allgather.py>`_

.. 3. [Communication]: `Inter-node AllGather <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/03-inter-node-allgather.py>`_

.. 4. [Communication]: `Intra-node and Inter-node DeepSeek EP AllToAll <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/04-deepseek-infer-all2all.py>`_

.. 5. [Communication]: `Intra-node ReduceScatter <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/05-intra-node-reduce-scatter.py>`_

.. 6. [Communication]: `Inter-node ReduceScatter <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/06-inter-node-reduce-scatter.py>`_

.. 7. [Overlapping]: `AllGather GEMM overlapping <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/07-overlapping-allgather-gemm.py>`_

.. 8. [Overlapping]: `GEMM ReduceScatter overlapping <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/08-overlapping-gemm-reduce-scatter.py>`_

.. 9. [Overlapping]: `AllGather GEMM overlapping on AMD <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/09-AMD-overlapping-allgather-gemm.py>`_

.. 10. [Overlapping]: `GEMM ReduceScatter overlapping on AMD <http://https://github.com/ByteDance-Seed/Triton-distributed/blob/main/tutorials/10-AMD-overlapping-gemm-reduce-scatter.py>`_
