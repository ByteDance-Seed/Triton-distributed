#include <driver_types.h>
#include <torch/all.h>

namespace distributed {

namespace ops {

void moe_ag_scatter_align_block_size_parallel_op(
    torch::Tensor topk_ids, int32_t topk, int32_t num_experts,
    int32_t num_ranks, int32_t num_tokens_per_rank, int32_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor block_barrier_ids, torch::Tensor rank_block_num,
    torch::Tensor num_tokens_post_pad, intptr_t moe_stream);

void moe_ag_scatter_align_block_size_op(
    torch::Tensor topk_ids, int32_t num_experts, int32_t num_iterations,
    int32_t num_tokens_per_iteration, int32_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor block_barrier_ids, torch::Tensor rank_block_num,
    torch::Tensor num_tokens_post_pad, intptr_t moe_stream);
} // namespace ops
} // namespace distributed