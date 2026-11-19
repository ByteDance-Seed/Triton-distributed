# Historical SM100 dataflow prototype reference

This document preserves prototype design and experiment context. Its old paths
and commands are historical references. The prototype directory has been removed;
use [the current dataflow guide](../../docs/dataflow.md) for supported APIs and
[verification](../../docs/verification.md) for maintained commands.

## Original prototype notes

This directory contains the SM100 role/dataflow compiler prototype. Kernel
source declares roles, logical memories, and RAW channels with
`dataflow.py`; compiler passes analyze those declarations and insert the
pipeline protocol.

## Source Model

```python
with dataflow() as g:
    load_a = g.role("load_a", hw=HW.TMA, warps=[0])
    load_b = g.role("load_b", hw=HW.TMA, warps=[1])
    mma = g.role("mma", hw=HW.TensorCore, warps=[2])
    a = g.smem("a", shape, dtype=dtype, stages=2)
    b = g.smem("b", shape, dtype=dtype, stages=2)
    stage = g.depends(src=[load_a, load_b], dst=mma, via=[a, b])

with load_a:
    while ...:
        stage.bind(i)
        ...
```

`depends` declares one RAW data channel. `src` and `dst` accept one role or a
list of roles, so a channel may be SPSC, SPMC, or MPMC. Every logical memory
has one channel owner. `stage.bind(value)` marks one logical transaction at
each endpoint; it accepts exactly one positional argument and must not appear
inside a runtime `if`.

The compiler compares endpoint bind streams for a restricted canonical loop
form and rejects unprovable or mismatched stream shapes. This is not yet a full
proof of cross-role runtime predicates inside a TMA-store transaction; the
remaining boundary is tracked in
[`docs/history/role-proposal/AUTO_PIPELINE_IMPLEMENTATION_REVIEW.zh.md`](../../../docs/history/role-proposal/AUTO_PIPELINE_IMPLEMENTATION_REVIEW.zh.md).
A logical MPMC channel is lowered to one runtime pipeline per producer; every
consumer waits and releases all of them.

Terminology is intentionally strict:

- **channel** means a source-level logical RAW relation declared by
  `g.depends`;
- **pipeline** means a lowered runtime synchronization object implementing a
  channel or a physical-reuse constraint.

Physical storage sharing is declared separately:

```python
g.memory_mapping_plan = [
    MemoryMappingEntry("logical_view", "physical_buffer", ...),
]
```

The overlay pass derives lifetime constraints and internal reuse pipelines
from this mapping, memory effects, channels, and existing bind regions. There
is no user-facing reuse channel.

## Compiler Pipeline

`pass_auto_pipe.py` runs:

1. compile-time unrolling;
2. UMMA mode/effect normalization;
3. dataflow, memory-effect, bind-region, ownership, stream, and TMA-partition
   validation;
4. structured-control-flow partitioning at logical-memory effect boundaries;
5. RMW-cycle analysis;
6. physical-buffer overlay analysis;
7. inferred overlay-pipeline construction;
8. per-domain TMA-store stream scheduling;
9. pipeline primitive insertion;
10. CUDA resource and intrinsic materialization.

Effect partitioning keeps producer acquisition early, but places publication
and consumption boundaries at the last write and first/last read. Uniform
two-arm branches receive branch-local boundaries; issuer/divergent guards are
split so pipeline participation remains unconditional. TMA-load regions remain
intact. TMA-store issuer guards may be split at source-lifetime frontiers so
independent stores become distinct hardware bulk groups.

Effect-pair pipeline selection is fail-closed. Intrinsics declare one normalized
effect per logical operand plus independent CTA-group topology;
`pipeline_templates.py` maps exact write-effect/read-effect pairs to a
`PipelineSpec`. The stable runtime header
[`little_kernel/backends/nv/sm100/runtime/include/role_pipeline.cuh`](../backends/nv/sm100/runtime/include/role_pipeline.cuh)
implements the runtime protocols, including empty TMA, UMMA, and TMA-store
transactions; the legacy `include/pipeline.cuh` is a thin forwarding include
until the role/dataflow promotion updates the generated `#include` path.
Each convergent warp domain has one statically scheduled TMA-store stream.
Payloads remain predicated, but every lane in the domain commits the group and
waits at the same completion frontier; non-issuing lanes therefore contribute
empty groups. The compiler maps each group to the source stages it reads and
emits literal `wait_group.read<N>` plus static per-pipeline retirement before
source reuse.
There are no runtime group IDs or owner records. Independent warp domains keep
independent streams and join through the channel empty barrier. TMA-store
pipeline depths are limited to the hardware maximum of eight.
TMA intrinsics carry explicit logical `smem_coord + shape`. Analysis validates
bounds and multi-producer coverage, derives expected transaction bytes, and
only then expands the region into canonical physical boxes. Device lowering
already shares one SMEM layout plan with UMMA descriptors; host `CUtensorMap`
metadata migration is tracked in the implementation review.

See [`docs/history/role-proposal/STAGE_REGION_FORMALISM.md`](../../../docs/history/role-proposal/STAGE_REGION_FORMALISM.md)
for bind-region rules and
[`docs/history/role-proposal/TMA_UMMA_SEMANTICS_DESIGN.zh.md`](../../../docs/history/role-proposal/TMA_UMMA_SEMANTICS_DESIGN.zh.md)
for the layout/descriptor contract.

## Intrinsic Layering

The public namespace is an analysis contract:

- `ll` contains thin physical wrappers: one PTX operation, or a narrow physical
  macro required by register-tuple / inline-assembly constraints;
- `ri` contains analyzable semantic operations over logical memories. A
  composite `ri` operation may lower to several `ll` operations, but every
  logical-memory operand must retain its access effect and recoverable region;
- kernels retain scheduling policy: role and stage boundaries, traversal
  order, issuer predicates, runtime control flow, and pipeline ownership.

`ll_kernel(is_entry=False)` is a device-function/code-reuse mechanism, not a
layering decision. An `ri` composite may contain the narrow wait/fence needed
to complete its own register result, but it must not hide compiler-owned
`producer_*` / `consumer_*` transitions or TMA-store stream retirement.
Intrinsic names use `ld` / `st` consistently and expose observable TMEM layout
and repeat, for example `tmem_ld_32dp_f32_to_smem_bf16_x8`; a future `16dp`
layout is a separate semantic operation rather than an implicit lowering choice.
Intrinsic analysis must be metadata-driven; adding name-specific cases to
`_logical_region()` is not the extension model. The stable intrinsic-layering
rule and current implementation status are captured in the campaign records
under [`docs/history/role-proposal/`](../../../docs/history/role-proposal/README.md).

## Main Files

- `dataflow.py`: role, logical-memory, channel, and stage-key declarations.
- `intrins.py`: role-proposal intrinsics and memory/pipeline metadata.
- `pass_auto_pipe.py`: graph analysis, pipeline inference, AST insertion, and
  CUDA materialization.
- `pass_channel.py`: MPMC channel classification, stream/ownership checks, and
  TMA payload validation.
- `pass_buffer_overlay.py`: physical placement, overlap, and lifetime analysis.
- `pass_rmw.py`: loop-carried RMW-cycle analysis.
- `pass_compile_time_unroll.py`: static loop expansion used before analysis.
- `pass_umma_modes.py`: UMMA init/accum and K-block effect normalization.
- `pipeline_templates.py`: supported compiler pipeline specifications.
- `include/pipeline.cuh`: SM100 pipeline runtime; the stable packaged copy is
  `little_kernel/backends/nv/sm100/runtime/include/pipeline.cuh`.
- The generated-kernel GEMM/FA4 examples move to `examples/role_proposal/`.
- The end-to-end SM100 correctness/performance runner and its canonical
  performance table live under `benchmarks/nv/common/role_proposal/`; the table must be
  compared and updated after material changes.
- Fail-closed channel/stream/TMA-partition/stage-overlay contract tests live
  under the canonical `tests/` tree.

## Verification

Run from the repository root using the committed `uv.lock` (Python 3.11+).
V2 GPU tests need an SM100 device and `nvcc` supporting `sm_100a`; CUDA 13.1
and four peer-accessible GB200 GPUs are the validated configuration. SM90 cubins
are not part of this GB200 acceptance run.

```bash
# Complete correctness + default-shape performance acceptance on four GB200s.
# The bench group installs optional reference helpers, not external source repos.
uv run --locked --group bench bash tools/nv/sm100/verify_v2.sh \
    --references required --check-perf --report /tmp/lk-v2-results.json

# Standalone kernels: no FLA/FlashAttention/FlashInfer/DeepGEMM/FlashComm required.
uv run --locked bash tools/nv/sm100/verify_v2.sh --references none --check-perf

# A single benchmark; auto uses available external references and reports omissions.
uv run --locked python -m benchmarks.role_proposal.bench_v2_gdn --references none
uv run --locked --group bench python -m benchmarks.role_proposal.bench_v2_fa4_sm100 --references required
```

`tools/nv/sm100/verify_v2.sh` is the local/CI acceptance entry point: unit tests, channel
contracts, SM100 GEMM integration, cluster topology, then all 13 V2 benchmarks.
Run it on the GPU worker; a Codebase “no pipeline configured” success is not test
evidence. The generated-kernel runner is a script and must be invoked explicitly,
not just collected with pytest.

`--references auto|required|none` applies to the external comparators. `auto`
prints `REFERENCE UNAVAILABLE` and keeps the LittleKernel check/measurement when
a reference cannot be imported. `required` fails on missing references; errors
while executing an imported reference always fail. `none` uses PyTorch checks
(SDPA for FA4, FP32 chunk recurrence for GDN, and native symmetric-memory routing
plus matmul/SwiGLU for dispatch). DeepGEMM has an independent PyTorch reference.
GDN checks every selected shape, including large_5x8k and throughput_512k.

All Python paths below accept colon-separated **import roots**. Explicit paths
precede installed packages. Otherwise an installed package is preferred, followed
by explicitly configured source roots. Empty path variables suppress the
source-path fallback; `--references none` disables imports altogether.

| Environment variable | Default source fallback |
|---|---|
| `FLA_PYTHON_PATH` | Installed package; `FLA_ROOT` can select an explicit checkout |
| `FLASH_ATTN_PYTHON_PATH` | Installed package or explicitly configured import roots |
| `FLASHINFER_PYTHON_PATH` | Installed package; `FLASHINFER_ROOT` can select an explicit checkout |
| `DEEPGEMM_PYTHON_PATH` | Installed package; `DEEPGEMM_ROOT` / `--deepgemm-root` can select a checkout and its `build/lib.*` directories |
| `FLASHCOMM_PYTHON_PATH` | sibling `Triton-distributed/FlashComm/python` |
| `CUTLASS_DSL_PYTHON_PATH` | `~/.local/lib/pythonX.Y/site-packages/nvidia_cutlass_dsl/python_packages`, then sibling `cutlass/python/CuTeDSL` |

For example:

```bash
FLA_PYTHON_PATH=/path/to/flash-linear-attention \
FLASHINFER_PYTHON_PATH=/path/to/flashinfer \
uv run --locked --group bench python -m benchmarks.role_proposal.bench_v2_gdn --references required
```

Build external native extensions (DeepGEMM / FlashComm) against the same Python,
PyTorch and CUDA ABI as this locked environment; setting a path does not rebuild
an incompatible `.so`. The optional `bench` group contains CuTeDSL, TVM FFI,
Quack and reference utilities. Core LittleKernel dependencies remain separate.
CuTeDSL and Quack are pinned together for the current FlashAttention
source API; upgrading either requires rerunning the reference benchmarks.
The distributed runner disables the other optional Triton-distributed and
ThunderKittens comparisons using their existing CLI flags.

Reference source revisions used for acceptance (native extensions must still be
built for the locked environment):

| Reference | Revision |
|---|---|
| Flash Linear Attention | `e47d5d20aeb5989b58a3738b872e7c288a9fb75f` |
| FlashAttention | `940cd9680f3315f2f06b43ab5bea2c2cf2d96806` |
| FlashInfer | `5366177a074e27df7db527f5b744c77dfd748484` |
| DeepGEMM | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` |
| Triton-distributed / FlashComm | `1b43bffce2711a1b089f0a737474c39bfb63d733` + patch below |

The existing FlashComm checkout has a scheduler constructor compatibility fix
for the pinned CuTeDSL API. It is included as `test/flashcomm-cutlass-compat.patch`
so a fresh reference checkout can reproduce it. Apply once from the LittleKernel
root (the default development checkout already has this fix):

```bash
git -C /path/to/Triton-distributed apply "$PWD/test/flashcomm-cutlass-compat.patch"
```

The fixed performance table lives in the generated-kernel benchmark under
`benchmarks/nv/common/role_proposal/`.
`--check-perf` parses the tracked default-shape **LittleKernel latency** and fails
above the recorded value plus `--perf-tolerance` (default 0.10). It requires GB200,
exactly four visible GPUs for distributed rows, and default warmup/iters. External
reference ratios are reported for comparison; they are not an automatic gate
because external implementations can change. `--require-all` fails on hardware
skips. A run without external references cannot reproduce the reference ratios.
Do not update baselines merely to make a regression pass.

The wheel must package the pipeline header as a runtime resource at
`little_kernel/backends/nv/sm100/runtime/include/pipeline.cuh` (its stable owner; during
promotion `role_proposal/include/pipeline.cuh` is the transitional facade).
Build and inspect with declared locked tooling:

```bash
uv run --locked uv build --wheel
uv run --locked pytest -q tests/repository/test_distribution.py
```

Source-tree/editable installs alone do not catch missing package data; the
installed-wheel check discovers the header through `importlib.resources` and
never reads repository example kernels.

## V2 API migration

Kernel launch topology uses `cluster_shape` and `cta_group_shape`; the old
`cluster_dim` keyword is removed. Intrinsics use `ld`/`st` naming and explicit
logical regions; copy new examples rather than mixing V1 intrinsic signatures
with V2. The role/dataflow layer remains a prototype with the compiler proof
boundaries documented in
[`docs/history/role-proposal/AUTO_PIPELINE_IMPLEMENTATION_REVIEW.zh.md`](../../../docs/history/role-proposal/AUTO_PIPELINE_IMPLEMENTATION_REVIEW.zh.md).
