.. dist-triton documentation master file, created by
   sphinx-quickstart on Tue May 27 11:17:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Triton-distributed's documentation!
==============================================

Triton-distributed is a distributed compiler designed for computation-communication overlapping, which is based on OpenAI Triton.

Using Triton-distributed, programmers are able to develop efficient kernels comparable to highly-optimized libraries (including `Distributed-GEMM <https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm>`_ and `FLUX <https://github.com/bytedance/flux/blob/main/README.md>`_). Triton-distributed currently mainly targets Nvidia GPU and AMD GPU. It can also be ported to other hardware platforms. Feel free to contact us if you want to use Triton-distributed on your own hardware.


Getting Started
---------------

- Follow the :doc:`installation instructions <getting-started/installation>` for your platform of choice.
- Take a look at the :doc:`tutorials <getting-started/tutorials/index>` to learn how to write your first Triton-distributed program.
- Explore our :doc:`end-to-end integration <getting-started/e2e/index>` to learn how Triton-Distributed accelerates inference for real-world LLMs.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/tutorials/index
   getting-started/e2e/index


Python API
----------

- :doc:`triton-dist.language <python-api/triton-dist.language>`
- :doc:`Triton-distributed semantics <python-api/triton-dist.semantics>`


.. toctree::
   :maxdepth: 1
   :caption: Python API
   :hidden:

   python-api/triton-dist.language
   python-api/triton-dist.semantics
