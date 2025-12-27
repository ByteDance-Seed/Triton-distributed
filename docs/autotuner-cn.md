# Triton-distributed 自动调优器

> **Language / 语言**: [English](autotuner.md) | [中文](autotuner-cn.md)

Triton-distributed 提供两种自动调优机制：

1. **`triton_dist.tune.autotune`** - 函数级自动调优器，用于调优带有配置空间的任意函数（推荐使用）
2. **`triton_dist.autotuner.contextual_autotune`** - 上下文自动调优器，用于分布式调优包含 `triton.autotune` 装饰器的函数

## 函数级自动调优器 (`triton_dist.tune.autotune`)

这是 Triton-distributed 中推荐的函数调优方式。它提供：

- 支持 `key_fn` 和 `prune_fn` 的配置空间
- 自动缓存调优结果到 `~/.triton_dist/autotune/`
- 硬件和软件版本跟踪
- 通过进程组支持分布式调优
- 基于共享内存等约束的自动配置裁剪

### 基本用法

```python
import triton
import triton_dist
from triton_dist.tune import autotune

# 定义配置空间
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

# 定义用于缓存的 key 函数
def key_fn(A, B, *args, **kwargs):
    return (A.shape, B.shape, A.dtype)

# 可选：定义裁剪函数以跳过无效配置
def prune_fn(config, A, B, *args, **kwargs):
    # 跳过超出共享内存的配置
    shared_mem = config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_K"] * A.element_size()
    return shared_mem < 48 * 1024  # 48KB 限制

@autotune(
    config_space=[{"gemm_config": c} for c in get_config_space()],
    key_fn=key_fn,
    prune_fn=prune_fn,
)
def my_gemm_function(A, B, gemm_config: triton.Config):
    # 你的函数实现
    ...
```

### 函数级自动调优器参数

```python
triton_dist.tune.autotune(
    config_space,    # 要调优的配置字典列表
    key_fn,          # 从参数生成缓存 key 的函数
    prune_fn=None,   # 可选的配置裁剪函数
)
```

**参数说明：**
- `config_space`：包含可调参数的字典列表
- `key_fn`：接受与被装饰函数相同参数的函数，返回用于缓存的可哈希 key
- `prune_fn`：可选函数，返回 `True` 表示配置有效，返回 `False` 跳过该配置

**调用自动调优函数：**

```python
# 启用自动调优的正常调用
result = my_gemm_function(A, B)

# 禁用自动调优（使用第一个配置）
result = my_gemm_function(A, B, autotune=False)

# 启用详细日志
result = my_gemm_function(A, B, autotune_verbose=True)

# 使用特定进程组进行分布式调优
result = my_gemm_function(A, B, autotune_pg=my_process_group)
```

### 实际例子：AllGather GEMM

来自 `python/triton_dist/kernels/nvidia/allgather_gemm.py`：

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
    # 裁剪超出共享内存的配置
    if not prune_fn_by_shared_memory(config, A, *args, **kwargs):
        return False
    # 裁剪不符合 group size 的配置
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
    """AllGather GEMM 实现"""
    # 实现细节...
    pass
```

### 缓存行为

自动调优器将结果缓存在 `~/.triton_dist/autotune/<function_name>/`：
- 缓存文件为 JSON 格式，包含硬件/软件版本跟踪
- 当硬件或软件版本变化时，结果会失效
- 设置 `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE=1` 可强制重新调优

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE` | `0` | 即使缓存存在也强制重新调优 |
| `TRITON_DIST_AUTOTUNE_VERSION_CHECK` | `0` | 严格版本检查 |

---

## 上下文自动调优器 (`triton_dist.autotuner.contextual_autotune`)

此自动调优器专为调优包含 `triton.autotune` 装饰的 Triton kernel 的函数设计。适用于以下场景：

1. 函数包含多个带有 `triton.autotune` 装饰器的 Triton kernel
2. Kernel 有副作用，无法单独调优
3. 调优过程中需要分布式同步

### 上下文自动调优器用法

```python
from triton_dist.autotuner import contextual_autotune

@contextual_autotune(is_dist=True, n_repeat=5, n_warmup=3)
def my_distributed_function():
    # 此函数包含 triton.autotune 装饰的 kernel
    ...
```

### 上下文自动调优器参数

```python
triton_dist.autotuner.contextual_autotune(
    is_dist=False,   # 启用分布式调优
    n_repeat=5,      # 每个配置的计时迭代次数
    n_warmup=3,      # 预热迭代次数
)
```

### 示例：带有 Triton Autotune 的 AllGather GEMM

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
    # 设置 tensor...
    
    @contextual_autotune(is_dist=True)
    def run_ag_gemm_persistent():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # 通信阶段
        local_copy_and_barrier_all(...)
        # 带有自动调优 kernel 的计算阶段
        ag_gemm_persistent(A, B, C, rank, num_ranks, ...)
        return C
    
    # 运行自动调优
    C = run_ag_gemm_persistent()
```

### 工作原理

1. `ContextualAutotuner` 拦截对 `triton.autotune` 装饰的 kernel 的调用
2. 它多次运行被装饰的函数，尝试不同的配置
3. 每个配置都会被测量，选择最佳配置
4. 在分布式模式下，结果会跨 rank 同步

**调优过程：**

| 调优迭代 | kernel-0 | kernel-1 |
|----------|----------|----------|
| 0 | config-0 (iter-0) | config-0 (iter-0) |
| 1 | config-0 (iter-1) | config-0 (iter-1) |
| 2 | config-1 (iter-0) | config-1 (iter-0) |
| 3 | config-1 (iter-1) | config-1 (iter-1) |
| 4 | **最佳配置** | config-2 (iter-0) |
| 5 | **最佳配置** | config-2 (iter-1) |
| 最终 | **最佳配置** | **最佳配置** |

日志保存在 `./.autotune_logs/rank-{i}.log`。

---

## 选择合适的自动调优器

| 使用场景 | 推荐 |
|----------|------|
| 调优带有配置空间的 Python 函数 | `triton_dist.tune.autotune` |
| 包含 `triton.autotune` kernel 的函数 | `triton_dist.autotuner.contextual_autotune` |
| 分布式 GEMM/通信 kernel | `triton_dist.tune.autotune` |
| 简单 Triton kernel 调优 | `triton.autotune`（原生 Triton） |

## 测试命令

```bash
# 使用函数级自动调优测试
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check

# 使用上下文自动调优测试
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_tma_autotune

# MoE 自动调优测试
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check --autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --autotune
```
