################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import torch
from flash_comm.ep.ep_kernels import EPCommLayoutDesc


def tensor(*shape):
    return torch.empty(shape, device="meta")


def expect_value_error(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def main():
    intranode = EPCommLayoutDesc(
        token_within_expert_offset=tensor(32, 8),
        expert_counts=tensor(161),
        recv_base_offset=tensor(8, 20, 8),
        token_dst_scatter_indices=tensor(32, 8),
        token_topk_send_mask=tensor(32, 8),
        topk_indices=tensor(32, 8),
        recv_token_count_cpu=tensor(8),
        recv_token_count=tensor(8),
        recv_expert_counts=tensor(20),
        num_tokens=32,
    )
    intranode.check_layout_desc(num_tokens=32, topk=8, num_experts=160, world_size=8)

    internode = EPCommLayoutDesc(
        token_dst_scatter_indices=tensor(32, 8),
        token_topk_send_mask=tensor(32, 8),
        node_topk_indices=tensor(2, 64, 8),
        node_topk_send_mask=tensor(2, 64, 8),
        node_token_dst_scatter_indices=tensor(2, 64, 8),
        num_tokens=32,
    )
    internode.check_layout_desc(
        num_tokens=32,
        topk=8,
        num_experts=160,
        world_size=8,
        local_world_size=4,
        max_slot_num_token=64,
    )
    internode.check_internode_combine_required_inputs()

    invalid_sender = EPCommLayoutDesc(token_dst_scatter_indices=tensor(2, 64, 8), num_tokens=32)
    expect_value_error(
        lambda: invalid_sender.check_layout_desc(
            num_tokens=32,
            topk=8,
            num_experts=160,
            world_size=8,
            local_world_size=4,
            max_slot_num_token=64,
        ),
        "token_dst_scatter_indices must have shape [32, 8]",
    )

    invalid_node_metadata = EPCommLayoutDesc(node_topk_indices=tensor(2, 64, 8))
    expect_value_error(
        lambda: invalid_node_metadata.check_layout_desc(
            num_tokens=32,
            topk=8,
            num_experts=160,
            world_size=8,
        ),
        "node_topk_indices is internode-only metadata",
    )

    assert not intranode.need_recompute_dispatch_layout(num_tokens=32)
    assert intranode.need_recompute_dispatch_layout(num_tokens=33)
    print("EPCommLayoutDesc stateless layout checks passed")


if __name__ == "__main__":
    main()
