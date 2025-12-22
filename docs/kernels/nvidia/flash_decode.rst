Flash Decode
============

Distributed Flash Decoding kernels for attention computation.

API Reference
-------------

.. py:function:: gqa_fwd_batch_decode(...)

   GQA forward batch decode kernel.

.. py:function:: gqa_fwd_batch_decode_persistent(...)

   Persistent version of GQA forward batch decode.

.. py:function:: gqa_fwd_batch_decode_aot(...)

   AOT-compiled GQA forward batch decode.

.. py:function:: gqa_fwd_batch_decode_persistent_aot(...)

   AOT-compiled persistent GQA forward batch decode.

.. py:function:: gqa_fwd_batch_decode_intra_rank(...)

   Intra-rank GQA forward batch decode.

.. py:function:: gqa_fwd_batch_decode_intra_rank_aot(...)

   AOT-compiled intra-rank GQA forward batch decode.

.. py:function:: kernel_gqa_fwd_batch_decode_split_kv_persistent(...)

   Persistent kernel for split KV flash decode.

.. py:function:: kernel_inter_rank_gqa_fwd_batch_decode_combine_kv(...)

   Inter-rank kernel for combining KV results.

.. py:function:: get_triton_combine_kv_algo_info(...)

   Gets algorithm info for KV combination.

Performance
-----------

Flash decode scales efficiently from 1 GPU to 32 GPUs with minimal latency overhead.

