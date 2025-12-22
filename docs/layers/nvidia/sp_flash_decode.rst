Sequence Parallel Flash Decode
==============================

Sequence Parallel Flash Decode layer for distributed attention decoding.

Description
-----------

This layer implements distributed flash attention decoding with sequence parallelism,
efficiently distributing the KV cache across multiple GPUs.

Example Usage
-------------

.. code-block:: bash

   # Test SP Flash Decode
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case perf

