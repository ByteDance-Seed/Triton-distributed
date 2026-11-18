# LittleKernel in Triton-distributed

The bundled compiler supports `nv/sm90`, `nv/sm100` and `amd/cdna4` (`gfx950`).
Shared types, IR, compiler sessions, passes, C++ visitors, ABI and binary-cache
services are composed with the selected backend's language extensions, codegen,
toolchain and runtime. Only these hardware implementations are bundled.

`upstream.json` records the source revision without importing another repository's
commit history. Compiler and runtime code is in `python/little_kernel`.
Tests, benchmarks, examples and reference assets stay outside the installed package.

## CPU validation

The compiler SDK and its locked check environment require Python 3.11 or newer.
From the repository root:

```bash
uv sync --locked --group cpu-test --group build --group format
uv run --locked pytest -q
uv run --locked pytest -q tests/repository
uv run --locked python tools/verify.py --list
uv run --locked python tools/verify.py --backend amd/cdna4 --suite cpu
```

The root UV project supplies CPU checks and packaging/format tools. The existing
`python/setup.py` remains the Triton-distributed packaging entry point. Its
`TRITON_BUILD_LITTLE_KERNEL` switch controls the bundled compiler, including its
NumPy dependency. All backend subpackages, descriptor files and required runtime
headers are included when that switch is enabled.

## Hardware validation

```bash
uv run --locked python tools/verify.py --backend nv/sm90 --suite gpu
uv run --locked python tools/verify.py --backend nv/sm100 --suite gpu
uv run --locked python tools/verify.py --backend amd/cdna4 --suite gpu
```

These commands need the corresponding device, driver, NVCC/HIPCC and tensor
integration. Add tensor/benchmark dependencies to the intended deployment's
dependency configuration and lock before running device suites. The CPU lock
does not install a GPU tensor stack. Missing hardware and all-skipped required
gates return `unavailable`; they never count as a device or performance pass.

## Layout

| Responsibility | Location |
| --- | --- |
| Shared compiler services | `python/little_kernel/{core,language,codegen,runtime}` |
| Backend contracts and discovery | `python/little_kernel/backends/{interfaces,registry,shared}.py` |
| NVIDIA common implementation | `python/little_kernel/backends/nv/common` |
| Hopper and Blackwell components | `python/little_kernel/backends/nv/{sm90,sm100}` |
| CDNA4 components | `python/little_kernel/backends/amd/cdna4` |
| Tests | `tests/{common,nv,amd,repository}` |
| Benchmarks and authored programs | `benchmarks/{nv,amd}`, `examples/nv` |
| Reference material | `assets/nv` |

The SM100 dataflow frontend has explicit language, protocol, pass and codegen
owners under `backends/nv/sm100`. Its pipeline header is an installed package
resource at `backends/nv/sm100/runtime/include/role_pipeline.cuh`.

## Additional selected source exports

```bash
uv run --locked python tools/export_source.py \
    --backends nv/sm90,amd/cdna4 --output /tmp/little-kernel-selected
cd /tmp/little-kernel-selected
uv sync --locked --group cpu-test --group build
uv run --locked pytest -q
uv run --locked uv build --wheel --no-build-isolation
```

The output must be empty or absent. Export discovery uses the physically bundled
backend descriptors and copies their dependency closure, selected programs/tests
and required resources. It has no fallback to another compiler checkout.
