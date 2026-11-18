# ai-compiler-eval bridge

The compiler consumes an agent-authored LittleKernel frontend program
(`draft.py`) and a launch/shape manifest (`export.json`). The same LittleKernel
IR, CUDA passes, code generator, and generic exporter compile every program.
The PyTorch `model.py` and `test_cases.py` belong to the trusted correctness
oracle; they are never parsed to select a task-specific CUDA template.

The bridge accepts `lower(task_dir, program_dir, output_dir, case_index)`.
It selects `program_dir/case-NNN/` when the case directory carries its own
`export.json` or `pipeline.json`, otherwise `program_dir/`, and emits
`model_new.py` plus `kernels/*.cu` and a C++ binding. A case directory without
`draft.py` compiles the shared top-level `draft.py` with the case's export
manifest, so per-case `constexpr` constructor specializations do not duplicate
frontend source. Agent programs may specialize dtypes/shapes by case. The
evaluator compares the generated output with the fixed PyTorch reference and
an independently measured best_impl.

The general exporter implementation lives in
`little_kernel.integrations.vibe_cuda_eval`. It compiles the annotated
`@lk.ll_kernel` source through LittleKernel's IR, passes, and CUDA codegen;
the generated submission has no LittleKernel runtime dependency.

For a reference `Model` with constructor arguments, declare them in
`export.json`, for example:

```json
"constructor": [{"name": "scale", "dtype": "float32", "default": 1.0}],
"kernel_args": ["x", "out", "n", "scale"]
```

The generated `Model.__init__` accepts the declared arguments and forwards
used arguments through the C++ binding to the LittleKernel entry. Supported
argument dtypes are `int32`, `int64`, `float32`, and `bool`. Use `ignored` for a
constructor argument that the case-specialized frontend does not need at
runtime. Integer constructor arguments can also appear in output shapes and
launch grid expressions.

A constructor argument can instead be baked into the compiled kernel as a
compile-time constant, the frontend analogue of a C++ template parameter:

```json
"constructor": [{"name": "page_size", "dtype": "int32", "constexpr": 64}],
"kernel_args": ["keys", "values", "out", "n", "page_size"]
```

`"constexpr"` supplies the literal used at trace time: every read of the argument
inside the entry function becomes a Python literal, so `range(page_size)` trip
counts, address expressions and unroll decisions specialize exactly like a hand
instantiated template. The argument is still declared in `Model.__init__` (a
mismatched runtime value raises `ValueError`), but it leaves the runtime kernel
ABI — the CUDA kernel, launcher and binding argument lists drop it. A generated
host-side constant keeps shape, scalar, grid and equality expressions that name
it working. A `constexpr` entry cannot also be `ignored` or declare a `default`,
and its name is read-only in the frontend (assigning to it is a compile-time
error). Because the exported spec records the literal, distinct per-case
`case-NNN/export.json` files compile one shared `draft.py` into distinct
submissions without duplicating the frontend source.

For an operator that needs more than one CUDA launch, use a declarative
`pipeline.json` instead of a top-level `draft.py` and `export.json`:

```json
{
  "inputs": ["x"],
  "outputs": ["out"],
  "stages": [{"name": "first"}, {"name": "second"}]
}
```

Put each stage's LittleKernel `draft.py` and `export.json` under
`stages/<name>/`. A stage's `inputs` may name model inputs or tensors produced
by earlier stages. Stage output names become available to later stages, and
the pipeline's `outputs` list selects the model return values. Stage function
names must be distinct. A pipeline may also declare `constructor` using the
same format as a single-stage program; stages declare any constructor
arguments they consume. The exporter compiles every stage through the same
LittleKernel IR, passes, and CUDA codegen and generates one model wrapper that
calls the stages in order.

Prefer a single stage when the task's CUDA submission launches one kernel.
The evaluator profiles actual kernel launches and rejects a generated program
that exceeds the original submission's count for the same case. Write a
module docstring describing the algorithm and short function docstrings for
important device routines; CUDA codegen preserves them as source comments.
