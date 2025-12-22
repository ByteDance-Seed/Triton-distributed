# AutoTuner for Triton-distributed

> **Language / 语言**: [English](autotuner.md) | [中文](autotuner-cn.md)

Triton-distributed provides two autotuning mechanisms:

1. **`triton_dist.tune.autotune`** - Function-level autotuner for tuning arbitrary functions with config spaces (recommended)
2. **`triton_dist.autotuner.contextual_autotune`** - Contextual autotuner for distributed tuning of functions containing `triton.autotune`-decorated kernels

## Function-Level AutoTuner (`triton_dist.tune.autotune`)

This is the recommended approach for tuning functions in Triton-distributed. It provides:

- Config space with `key_fn` and `prune_fn` support
- Automatic caching of tuning results to `~/.triton_dist/autotune/`
- Hardware and software version tracking
- Distributed tuning via process groups
- Automatic config pruning based on shared memory and other constraints

### Basic Usage

```python
import triton
import triton_dist
from triton_dist.tune import autotune

# Define config space
def get_config_space():
    return [
        triton.Config({
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": 8,
        }, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [128, 256]
        for BK in [32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]

# Define key function for caching
def key_fn(A, B, *args, **kwargs):
    return (A.shape, B.shape, A.dtype)

# Optional: Define prune function to skip invalid configs
def prune_fn(config, A, B, *args, **kwargs):
    # Skip configs that exceed shared memory
    shared_mem = config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_K"] * A.element_size()
    return shared_mem < 48 * 1024  # 48KB limit

@autotune(
    config_space=[{"gemm_config": c} for c in get_config_space()],
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def my_gemm_function(A, B, gemm_config: triton.Config):
    # Your function implementation
    ...
```

### Function-Level AutoTuner Parameters

```python
triton_dist.tune.autotune(
    config_space,    # List of config dicts to tune over
    key_fn,          # Function to generate cache key from args
    prune_fn=None,   # Optional function to prune invalid configs
)
```

**Parameters:**
- `config_space`: List of dictionaries containing tunable parameters
- `key_fn`: Function that takes the same arguments as the decorated function and returns a hashable key for caching
- `prune_fn`: Optional function that returns `True` if a config is valid, `False` to skip it

**Calling the autotuned function:**

```python
# Normal call with autotuning enabled
result = my_gemm_function(A, B)

# Disable autotuning (use first config)
result = my_gemm_function(A, B, autotune=False)

# Enable verbose logging
result = my_gemm_function(A, B, autotune_verbose=True)

# Use specific process group for distributed tuning
result = my_gemm_function(A, B, autotune_pg=my_process_group)
```

### Real-World Example: AllGather GEMM

From `python/triton_dist/kernels/nvidia/allgather_gemm.py`:

```python
import triton
import triton_dist
from triton_dist.tune import to_hashable

def ag_gemm_config_space():
    if is_cuda() and _is_hopper():
        return [{"gemm_config": x} for x in get_config_space(True)]
    else:
        return [{"gemm_config": x} for x in get_config_space(False)]

def key_fn(A, B, ctx, *args, **kwargs):
    return (to_hashable(A), to_hashable(B), ctx.num_ranks, ctx.local_num_ranks)

def prune_fn(config, A, B, ctx, *args, **kwargs):
    gemm_config = config["gemm_config"]
    # Prune configs that exceed shared memory
    if not prune_fn_by_shared_memory(config, A, *args, **kwargs):
        return False
    # Prune configs that don't fit the group size
    if not prune_fn_by_group_size_m(config, A, B, *args, **kwargs):
        return False
    return True

@triton_dist.tune.autotune(
    config_space=ag_gemm_config_space(),
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def ag_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    ctx: AllGatherGEMMTensorParallelContext,
    gemm_config: triton.Config,
    straggler_option=None,
):
    """AllGather GEMM implementation"""
    # Implementation details...
    pass
```

### Caching Behavior

The autotuner caches results in `~/.triton_dist/autotune/<function_name>/`:
- Cache files are JSON format with hardware/software version tracking
- Results are invalidated when hardware or software versions change
- Set `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE=1` to force re-tuning

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE` | `0` | Force re-tuning even if cache exists |
| `TRITON_DIST_AUTOTUNE_VERSION_CHECK` | `0` | Strict version checking |

---

## Contextual AutoTuner (`triton_dist.autotuner.contextual_autotune`)

This autotuner is designed for tuning functions that contain `triton.autotune`-decorated Triton kernels. It's useful when:

1. A function contains multiple Triton kernels with `triton.autotune` decorators
2. The kernels have side effects and cannot be tuned individually
3. Distributed synchronization is needed during tuning

### Contextual AutoTuner Usage

```python
from triton_dist.autotuner import contextual_autotune

@contextual_autotune(is_dist=True, n_repeat=5, n_warmup=3)
def my_distributed_function():
    # This function contains triton.autotune-decorated kernels
    ...
```

### Contextual AutoTuner Parameters

```python
triton_dist.autotuner.contextual_autotune(
    is_dist=False,   # Enable distributed tuning
    n_repeat=5,      # Number of timing iterations per config
    n_warmup=3,      # Number of warmup iterations
)
```

### Example: AllGather GEMM with Triton Autotune

```python
import triton
import triton_dist
from triton_dist.autotuner import contextual_autotune

def matmul_get_configs():
    return [
        triton.Config({
            "BLOCK_SIZE_M": BM,
            "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK,
            "GROUP_SIZE_M": 8,
        }, num_stages=s, num_warps=w)
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]

@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
@triton_dist.jit
def kernel_consumer_gemm_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr, comm_buf_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    ...

def test_ag_gemm(rank, num_ranks, default_group):
    # Setup tensors...
    
    @contextual_autotune(is_dist=True)
    def run_ag_gemm_persistent():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # Communication phase
        local_copy_and_barrier_all(...)
        # Computation phase with autotuned kernel
        ag_gemm_persistent(A, B, C, rank, num_ranks, ...)
        return C
    
    # Run with autotuning
    C = run_ag_gemm_persistent()
```

### How It Works

1. `ContextualAutotuner` intercepts calls to `triton.autotune`-decorated kernels
2. It runs the decorated function multiple times, trying different configurations
3. Each configuration is measured and the best one is selected
4. Results are synchronized across ranks in distributed mode

**Tuning Process:**

| Tuning-Iter | kernel-0 | kernel-1 |
|-------------|----------|----------|
| 0 | config-0 (iter-0) | config-0 (iter-0) |
| 1 | config-0 (iter-1) | config-0 (iter-1) |
| 2 | config-1 (iter-0) | config-1 (iter-0) |
| 3 | config-1 (iter-1) | config-1 (iter-1) |
| 4 | **best-config** | config-2 (iter-0) |
| 5 | **best-config** | config-2 (iter-1) |
| final | **best-config** | **best-config** |

Logs are saved to `./.autotune_logs/rank-{i}.log`.

---

## Choosing the Right AutoTuner

| Use Case | Recommended |
|----------|-------------|
| Tuning Python functions with config spaces | `triton_dist.tune.autotune` |
| Functions containing `triton.autotune` kernels | `triton_dist.autotuner.contextual_autotune` |
| Distributed GEMM/Communication kernels | `triton_dist.tune.autotune` |
| Simple Triton kernel tuning | `triton.autotune` (vanilla Triton) |

## Test Commands

```bash
# Test with function-level autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check

# Test with contextual autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_tma_autotune

# MoE tests with autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check --autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --autotune
```
