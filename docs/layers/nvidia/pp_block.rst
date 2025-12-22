Pipeline Parallel Block
=======================

High-level abstraction for Pipeline Parallelism.

Description
-----------

This module provides building blocks for implementing pipeline parallelism in large models.

Example Usage
-------------

.. code-block:: bash

   # Test PP Block
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_pp_block.py \
       --bsz 8 --seq_len 128 --num_blocks 4 --pp_size 4 --model <model_path>

