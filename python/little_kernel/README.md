# LittleKernel Python package

LittleKernel lowers decorated Python functions through shared compiler infrastructure
and hardware-specific NVIDIA CUDA/PTX and AMD HIP/AMDGPU components.

## Minimal example

```python
import torch

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends import get_backend
from little_kernel.core.passes import PASSES

codegen_cuda = get_backend("nv/sm90").codegen.generate

@lk.ll_kernel(backend="nv/sm90", is_entry=True)
def vec_add(
    a: ll.Tensor[ll.float32],
    b: ll.Tensor[ll.float32],
    out: ll.Tensor[ll.float32],
    n: ll.int32,
) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = a[i] + b[i]


n = 1024
kernel = vec_add.build(
    PASSES["nv/sm90"],
    codegen_cuda,
    grid=((n + 255) // 256, 1, 1),
    block=(256, 1, 1),
    arch="sm_90a",
)
a = torch.randn(n, device="cuda")
b = torch.randn(n, device="cuda")
out = torch.empty_like(a)
kernel(a, b, out, n)
kernel.synchronize()
```

## Control flow: runtime loops and loop-carried scalars

The frontend pipeline is AST -> passes -> codegen (see `core/compile.py`); the
Python function body is translated, not executed, so there is no SSA
restriction on scalars.

* `for i in range(start, stop, step)` with tensor-derived (runtime) bounds
  lowers to a CUDA `for` loop. `ll.unroll(range(...))` with constant bounds
  unrolls at trace time instead.
* **Loop-carried mutable scalars are supported and stable.** A scalar
  annotated once ABOVE the loop and reassigned inside the body (plain `=`,
  augmented `+=`, or conditionally under a nested runtime `if`) is emitted
  exactly once as a C++ declaration before the loop and mutated in place on
  every iteration. This is the idiom for software-pipelined/prefetch
  formulations, e.g. resolve iteration `i+stride`'s indices while computing
  iteration `i`:

  ```python
  nxt: ll.int32 = ll.int32(src[tid])          # carried prologue
  total: ll.int32 = 0                          # carried accumulator
  for i in range(tid, n, stride):
      cur: ll.int32 = nxt                       # loop-local temp (annotated)
      ahead: ll.int32 = i + stride
      if ahead < n:
          nxt = ll.int32(src[ahead])            # plain reassignment carries
          total += 1
      dst[i] = cur + total
  ```

* Reassignment sites must be PLAIN assignments — never re-annotate the
  carried name inside the loop (`nxt: ll.int32 = ...` would shadow it and
  break the carry). Loop-local temporaries, conversely, ARE annotated inside
  the body; they are hoisted once as uninitialized declarations and keep
  per-iteration semantics.
* `launch.disable_var_hoisting: true` (export.json) changes only where
  loop-local temporaries are declared; annotations above the loop keep
  carried semantics identically.

Emission contracts for all of the above are pinned by
`tests/nv/sm90/unit/test_loop_carried_scalars.py`.

## Package and hardware layout

Shared services live in `core/`, `language/`, `codegen/` and `runtime/`.
`backends/interfaces.py` defines common contracts. Hardware-owned extensions,
passes, codegen, runtime and integrations live in `backends/nv/{common,sm90,sm100}`
and `backends/amd/cdna4`. See the
[LittleKernel guide](../../docs/little_kernel/README.md).

Repository tests, examples, benchmarks and assets mirror hardware ownership and
stay outside the production wheel. `tests/` is the sole test root; there is no
prototype compatibility package. CPU compilation requires NumPy. Device/tensor
and benchmark integrations are optional, declared deployment dependencies.

## Verification

```bash
uv run --locked pytest -q tests
uv run --locked python tools/verify.py --backend nv/sm90 --suite cpu
uv run --locked python tools/verify.py --backend nv/sm90 --suite gpu
uv run --locked --group bench python -m benchmarks.nv.common.run_all --list
```

The GPU command requires matching hardware and toolkit. For SM100 dataflow, see
the [SM100 dataflow guide](../../docs/little_kernel/dataflow.rst). Selected hardware publishing is
described in the [LittleKernel documentation](../../docs/little_kernel/README.md).
