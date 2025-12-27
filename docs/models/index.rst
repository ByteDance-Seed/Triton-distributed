Models
======

Triton-distributed provides end-to-end model implementations with distributed inference support.

Model List
----------

.. list-table:: Available Models
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - :doc:`dense`
     - Dense transformer model (e.g., Qwen, LLaMA)
   * - :doc:`qwen_moe`
     - Qwen MoE model

.. toctree::
   :maxdepth: 1
   :hidden:

   dense
   qwen_moe
   engine
   config
   kv_cache

