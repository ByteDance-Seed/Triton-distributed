.. _sphx_glr_getting-started_tutorials_04-deepseek-infer-all2all.rst:

Low Latency All-to-All Communication
====================================
In this tutorial, we demonstrate how to implement the All-to-All communication
paradigm in Expert Parallelism (EP) for MoE models using Triton-distributed.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh tutorials/04-deepseek-infer-all2all.py


Motivations
-----------

First, let's quickly review the EP workflow:
In MoE, the `E` experts are distributed across `N` devices (EP ranks).
For simplicity, we assume that `N` divides `E` evenly, so experts are distributed
uniformly. For example, when `E = 128` and `N = 32`, each device will handle 4 experts.

During inference with EP, each device is assigned a subset of tokens, as determined
by the MoE router module. The router on each device generates a tensor of
shape `[num_tokens, topk]`, containing the indices of the top `k` experts selected
for each token. The experts chosen for a token may reside on other devices,
necessitating communication to send the tokens to the appropriate devices.
Similarly, if other devices have tokens that select experts located on the current device,
those tokens need to be sent to the current device as well. This process is called Dispatch.

After the tokens are processed by their corresponding experts, they need to be
returned to their original devices. This operation mirrors Dispatch and is referred
to as Combine. From a communication perspective, both Dispatch and Combine are
essentially All-to-All collective communication operations.

Next, we demonstrate how to implement an efficient All-to-All operation in Triton-distributed with minimal code.

Triton-distributed provides a programming model that allows fine-grained control
over data movement between devices, optimizing hardware utilization.
At the core of our implementation are low-level primitives that manage the communication logic.


Kernel
------

.. code-block:: Python

    @triton_dist.jit
    def all_to_all_kernel(
        send_tensor,
        data_src,
        data_dst,
        scale_src,
        scale_dst,
        splits_src,
        splits_dst,
        signal,
        send_splits_cumsum,
        recv_offset,
        rank: int,
        call_count: int,
        act_pos: int,
        MODE: tl.constexpr,
        ONLINE_QUANT_FP8: tl.constexpr,
        FP8_GSIZE: tl.constexpr,
        WORLD_SIZE: tl.constexpr,
        HIDDEN: tl.constexpr,
        MAX_M: tl.constexpr,
        NUM_TOT_EXPERTS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        """
        All-to-All kernel for the Dispatch and Combine phases.

        - send_tensor: The tokens to be sent.
        - data/scale/splits_src/dst: The source and destination symmetric buffers for communication.
        - signal: signal buffer for communication.
        - send_splits_cumsum: Cumulative sum of the token splits (expert-level) for the current rank.
        - recv_offset: only used in combine mode, the base offset of the received tokens.
        - call_count: as the unique ID used for signal operation.
        - act_pos: The position of the active buffer (0 or 1) for double buffering.

        - MODE: Determines whether the operation is Dispatch (0) or Combine (1).
        - ONLINE_QUANT_FP8: A flag indicating whether FP8 quantization is used.
        - FP8_GSIZE: The group size for FP8 quantization.
        - WORLD_SIZE: number of EP ranks.
        - HIDDEN: The hidden size for each token.
        - MAX_M: The maximum number of tokens that can be processed per rank.
        - EXPERTS_PER_RANK: The number of experts handled by each rank.
        - NUM_TOT_EXPERTS: The total number of experts.
        - BM, BN: Block size used to copy data to send buffer
        """
        pid = tl.program_id(0)
        # Triton-distributed exposes `tid` that can be used to identify the thread index within a CTA
        threadidx = tid(axis=0)
        NUM_GROUPS: tl.constexpr = HIDDEN // FP8_GSIZE
        EXPERTS_PER_RANK: tl.constexpr = NUM_TOT_EXPERTS // WORLD_SIZE


Calculate the token range for the current program (rank), get the corresponding pointer.

.. code-block:: Python

    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK
    m_st = tl.load(send_splits_cumsum + exp_st)
    m_ed = tl.load(send_splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    # Signal pointer to communicate when data is ready
    signal_ptr = signal + act_pos * WORLD_SIZE + rank
    if MODE == 0:  # dispatch mode
        # Calculate source and destination offsets based on the expert-level token number cumsum
        split_src_ptr = splits_src + (exp_st + pid)
        split_dst_ptr = splits_dst + act_pos * (NUM_TOT_EXPERTS + WORLD_SIZE) + rank * (EXPERTS_PER_RANK + 1)

        off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
        off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
        cumsum_sts = tl.load(send_splits_cumsum + off0)
        cumsum_eds = tl.load(send_splits_cumsum + off1)
        tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)
        tl.store(split_src_ptr + EXPERTS_PER_RANK, m_st)

        # Calculate the source and destination data offsets for the dispatch operation
        src_off = m_st
        dst_off = rank * MAX_M
        data_src_ptr = data_src + src_off * HIDDEN
        data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
        scale_src_ptr = scale_src + src_off * NUM_GROUPS
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M * NUM_GROUPS + dst_off * NUM_GROUPS
    else:  # combine mode
        # For the combine phase, source and destination offsets are updated accordingly
        src_off = pid * MAX_M
        dst_off = tl.load(recv_offset + pid)
        data_src_ptr = data_src + act_pos * WORLD_SIZE * MAX_M * HIDDEN + src_off * HIDDEN
        data_dst_ptr = data_dst + dst_off * HIDDEN
        scale_src_ptr = scale_src + act_pos * WORLD_SIZE * MAX_M * NUM_GROUPS + src_off * NUM_GROUPS
        scale_dst_ptr = scale_dst + dst_off * NUM_GROUPS


Copy the data (may be online quantized to FP8) to send buffer.

.. code-block:: Python

    off_m = tl.arange(0, BM)
    if ONLINE_QUANT_FP8 and MODE == 0:
        # TODO: adaptive UNROLL_FACTOR
        UNROLL_FACTOR: tl.constexpr = 4
        group_offs = off_m[:, None] * HIDDEN + tl.arange(0, FP8_GSIZE * UNROLL_FACTOR)[None, :]
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + group_offs
        data_src_ptrs = tl.cast(data_src_ptr, tl.pointer_type(tl.float8e4nv)) + group_offs
        scale_src_ptrs = scale_src_ptr + off_m[:, None] * NUM_GROUPS + tl.arange(0, UNROLL_FACTOR)[None, :]
        # online quant the input data to FP8
        for i in tl.range(ceil_div(num_rows_cur_block, BM)):
            group_mask = off_m[:, None] < num_rows_cur_block - i * BM
            for _ in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
                group = tl.reshape(tl.load(send_tensor_ptrs, group_mask), (BM * UNROLL_FACTOR, FP8_GSIZE))
                scale = tl.max(tl.abs(group), 1, keep_dims=True).to(tl.float32) * FP8_MAX_INV
                quant = tl.reshape((group.to(tl.float32) / scale).to(tl.float8e4nv), (BM, UNROLL_FACTOR * FP8_GSIZE))
                tl.store(data_src_ptrs, quant, group_mask)
                tl.store(scale_src_ptrs, tl.reshape(scale, (BM, UNROLL_FACTOR)), group_mask)
                send_tensor_ptrs += UNROLL_FACTOR * FP8_GSIZE
                data_src_ptrs += UNROLL_FACTOR * FP8_GSIZE
                scale_src_ptrs += UNROLL_FACTOR
            send_tensor_ptrs += (BM - 1) * HIDDEN
            data_src_ptrs += (BM - 1) * HIDDEN
            scale_src_ptrs += (BM - 1) * NUM_GROUPS
    else:
        off_n = tl.arange(0, BN)
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + off_m[:, None] * HIDDEN + off_n[None, :]
        data_src_ptrs = data_src_ptr + off_m[:, None] * HIDDEN + off_n[None, :]
        for i in tl.range(ceil_div(num_rows_cur_block, BM)):
            data_mask = (off_m[:, None] < num_rows_cur_block - i * BM) & (off_n[None, :] < HIDDEN)
            tl.store(data_src_ptrs, tl.load(send_tensor_ptrs, data_mask), data_mask)
            send_tensor_ptrs += BM * HIDDEN
            data_src_ptrs += BM * HIDDEN

Perform the memory copy operation using shared memory for inter-rank communication.

.. code-block:: Python

    # the last argument is the peer id (id of target rank)
    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src_ptr,
        num_rows_cur_block * HIDDEN * (1 if (ONLINE_QUANT_FP8 and MODE == 0) else 2),
        pid,
    )
    if MODE == 0:
        # Dispatch mode: send split information to the target rank
        libshmem_device.putmem_nbi_block(
            split_dst_ptr,
            split_src_ptr,
            (EXPERTS_PER_RANK + 1) * 4,  # now we use `int32` for splits
            pid,
        )
    # If online quantization is enbaled, signal the target rank with the scale data
    if ONLINE_QUANT_FP8:
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src_ptr,
            num_rows_cur_block * NUM_GROUPS * 4,  # assume `float32` for scale
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )
    
Fence data transfer. Then wait for signal

.. code-block:: Python

    libshmem_device.fence()
    if threadidx == 0:
        # notify the target rank (here is the `pid`-th rank) that the data is ready by setting the signal
        if not ONLINE_QUANT_FP8:
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        # wait for the signal from the source rank (here is the `pid`-th rank)
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )