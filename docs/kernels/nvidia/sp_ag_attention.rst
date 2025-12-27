Sequence Parallel AllGather Attention
=====================================

Fused Sequence Parallel AllGather + Attention kernels.

API Reference
-------------

Intra-Node
^^^^^^^^^^

.. py:function:: fused_sp_ag_attn_intra_node(...)

   Fused SP AllGather + Attention for intra-node communication.

.. py:function:: create_sp_ag_attention_context_intra_node(...)

   Creates context for intra-node SP AG Attention.

Inter-Node
^^^^^^^^^^

.. py:function:: fused_sp_ag_attn_inter_node(...)

   Fused SP AllGather + Attention for inter-node communication.

.. py:function:: create_sp_ag_attention_context_inter_node(...)

   Creates context for inter-node SP AG Attention.

Example Usage
-------------

.. code-block:: bash

   # Test SP AG Attention (intra-node)
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_ag_attention_intra_node.py \
       --batch_size 1 --q_head 32 --kv_head 32 --max_seqlen_q 8192 --max_seqlen_k 8192 --head_dim 128

