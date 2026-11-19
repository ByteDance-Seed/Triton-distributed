.. dist-triton documentation master file, created by
   sphinx-quickstart on Tue May 27 11:17:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Triton-distributed's documentation!
==============================================

``Triton-distributed`` is a distributed GPU compiler and kernel stack designed for
computation-communication overlapping across modern multi-GPU systems.

The repository brings together three tightly integrated components:

- **LittleKernel** (:doc:`little_kernel/index`) — Our flagship Python GPU kernel
  DSL and compiler targeting **SM90** (``nv/sm90``), **SM100** (``nv/sm100``), and
  **GFX950** (``amd/cdna4``), featuring explicit ``WGMMA`` / ``UMMA`` / ``MFMA`` and
  ``TMA`` / global-to-LDS DMA control as well as an automatic **Role & Dataflow**
  pipeline compiler on **SM100**.
- **FlashComm** (:doc:`flashcomm/index`) — A high-performance Expert-Parallel (EP)
  communication and compute-communication overlap library for **SM90** and **SM100**
  supporting intra-node NVLink, cross-node multi-NIC RDMA/GIN, fused ``MXFP8``
  dispatch, fixed-buffer chunked EP, and CuTeDSL fused ``Dispatch + GroupGEMM`` /
  ``GroupGEMM + Combine`` MegaKernels.
- **Triton-distributed Plugin & Primitives** (:doc:`primitives`) — An out-of-tree
  distributed compiler plugin on **Triton 3.7** together with ready-to-use NVIDIA
  and AMD distributed kernels, layers, and end-to-end LLM inference demos.


Getting Started
---------------

- Explore :doc:`LittleKernel <little_kernel/index>` and its :doc:`SM90 / SM100 performance benchmarks <little_kernel/performance>`.
- Explore :doc:`FlashComm <flashcomm/index>` for Expert-Parallel communication and compute-communication overlap on **SM90** and **SM100**.
- Follow the :doc:`build instructions <build>` for your platform of choice.
- Take a look at the :doc:`tutorials <getting-started/tutorials/index>` to learn how to write your first Triton-distributed program.
- Explore our :doc:`end-to-end integration <getting-started/e2e/index>` and :doc:`megakernel implementations <getting-started/megakernel/index>`.
- Learn how to :doc:`run all tests <testing>` to verify your installation.

.. toctree::
   :maxdepth: 10
   :caption: LittleKernel
   :hidden:

   little_kernel/index


.. toctree::
   :maxdepth: 10
   :caption: FlashComm
   :hidden:

   flashcomm/index


.. toctree::
   :maxdepth: 10
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/tutorials/index
   getting-started/e2e/index
   getting-started/megakernel/index
   getting-started/profiler/intra_kernel_profiler
   build
   testing
   primitives
   autotuner


Distributed Kernels
-------------------

Triton-distributed provides optimized distributed kernels for both NVIDIA and AMD GPUs:

- :doc:`kernels/index`
- :doc:`flashcomm/index`

Layer Abstractions
------------------

High-level layer abstractions for easier model integration:

- :doc:`layers/index`

Model Implementations
---------------------

End-to-end model implementations with distributed inference support:

- :doc:`models/index`


Python API
----------

- :doc:`triton-dist.language <python-api/triton-dist.language>`
- :doc:`Triton-distributed semantics <python-api/triton-dist.semantics>`


References
----------

- **UniEP (HPDC '26)** — `UniEP: Unified Expert-Parallel MegaKernel MoE for LLM Training <https://dl.acm.org/doi/10.1145/3806645.3807818>`_
- **DITRON (ICML 2026)** — `DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs <https://icml.cc/virtual/2026/poster/63503>`_
- **Triton-distributed (arXiv 2025)** — `Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler <https://arxiv.org/abs/2504.19442>`_
- **TileLink (MLSys 2025)** — `TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives <https://arxiv.org/abs/2503.20313>`_


.. toctree::
   :maxdepth: 10
   :caption: Kernels & Layers
   :hidden:

   kernels/index
   layers/index
   models/index


.. toctree::
   :maxdepth: 10
   :caption: Python API
   :hidden:

   python-api/triton-dist.language
   python-api/triton-dist.semantics


.. toctree::
   :maxdepth: 10
   :caption: Advanced Topics
   :hidden:

   amd-perf
   prepare_nvshmem
   e2e
   update-upstream


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/GALLERY_HEADER


.. toctree::
   :hidden:

   README
   autotuner-cn
