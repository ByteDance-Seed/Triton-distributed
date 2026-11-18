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
"""Generic ai-compiler-eval ABI for agent-authored LittleKernel programs.

The task's PyTorch reference is an oracle. The editable program is a separate
LittleKernel frontend source plus an export manifest. This module never
dispatches on a task name or inspects a PyTorch AST to choose CUDA text.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from little_kernel.backends.nv.common.export import check_identifier as _identifier
from little_kernel.backends.nv.common.export import export, load_spec


def _pipeline(selected: Path, manifest: Path, output_dir: Path, case_index: int) -> dict:
    """Compile a declarative sequence of authored LittleKernel stages."""
    pipeline = json.loads(manifest.read_text())
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline.json must contain an object")
    inputs = pipeline.get("inputs")
    outputs = pipeline.get("outputs")
    stages = pipeline.get("stages")
    constructor = pipeline.get("constructor", [])
    if (not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or not outputs
            or not isinstance(stages, list) or not stages or not isinstance(constructor, list)):
        raise ValueError("pipeline needs inputs, outputs, and stages lists")
    # Pipeline inputs may be plain tensor names or projections out of a
    # container forward argument: {"from": "cache", "key": "ck"} declares a
    # tensor named "ck" (override with "name") whose stage call-site spelling
    # is cache["ck"], while the generated Model.forward keeps one parameter
    # per container. Container entries are opt-in; plain-name pipelines are
    # unchanged.
    forward_params = []
    container_args = []
    arg_spelling = {}
    tensor_inputs = []
    for entry in inputs:
        if isinstance(entry, str):
            name = _identifier(entry, "pipeline input")
            forward_params.append(name)
            arg_spelling[name] = name
        elif isinstance(entry, dict):
            container = _identifier(entry.get("from"), "pipeline input container")
            key = entry.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("pipeline input container entry needs a string key")
            name = _identifier(entry.get("name", key), "pipeline input name")
            if container not in container_args:
                container_args.append(container)
                forward_params.append(container)
            arg_spelling[name] = f"{container}[{json.dumps(key)}]"
        else:
            raise ValueError("pipeline inputs must be names or container projections")
        tensor_inputs.append(name)
    inputs = tensor_inputs
    outputs = [_identifier(name, "pipeline output") for name in outputs]
    runtime_scalars = pipeline.get("runtime_scalars", [])
    if not isinstance(runtime_scalars, list):
        raise ValueError("pipeline runtime_scalars must be a list")
    runtime_scalars = [_identifier(name, "pipeline runtime scalar") for name in runtime_scalars]
    if any(name in inputs or name in outputs for name in runtime_scalars):
        raise ValueError("runtime scalar names must differ from pipeline inputs and outputs")
    if len(set(runtime_scalars)) != len(runtime_scalars):
        raise ValueError("pipeline runtime scalar names must be unique")
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError("pipeline inputs and outputs must be unique")
    if any(name in inputs or name in outputs or name in runtime_scalars for name in container_args):
        raise ValueError("pipeline input container names must differ from "
                         "declared inputs, outputs, and runtime scalars")
    # ignored_forward_args (incoming branch): ABI slots the generated
    # Model.forward accepts and drops (never forwarded to any stage). They
    # sit between the input/container slots and the runtime scalars in the
    # default signature; with forward_args they trail the fixed permutation.
    ignored = pipeline.get("ignored_forward_args", [])
    if not isinstance(ignored, list):
        raise ValueError("pipeline ignored_forward_args must be a list")
    ignored = [_identifier(name, "pipeline ignored forward argument") for name in ignored]
    if (any(name in inputs or name in outputs or name in runtime_scalars or name in container_args for name in ignored)
            or len(set(ignored)) != len(ignored)):
        raise ValueError("pipeline ignored forward arguments must be unique and "
                         "differ from declared inputs, outputs, containers, and scalars")
    passthrough = pipeline.get("passthrough_forward_args", [])
    if not isinstance(passthrough, list):
        raise ValueError("pipeline passthrough_forward_args must be a list")
    passthrough_names = []
    for item in passthrough:
        # ABI slots the generated Model.forward must accept and drop (e.g.
        # string selectors); they are never forwarded to any stage.
        name = _identifier(item, "pipeline passthrough forward argument")
        if name in inputs or name in outputs or name in passthrough_names:
            raise ValueError(f"pipeline passthrough forward argument must be a new unique name: {name}")
        passthrough_names.append(name)
    # passthrough slots live inside the forward_args permutation while
    # ignored slots are appended: an overlap would generate a duplicate
    # forward parameter and fail as a SyntaxError, so reject it here.
    if any(name in passthrough_names for name in ignored):
        raise ValueError("pipeline ignored and passthrough forward arguments must not overlap")
    forward_args = pipeline.get("forward_args")
    if forward_args is not None:
        if container_args:
            # Same rule as vibe_cuda_eval validation: a fixed positional
            # permutation cannot express container projections (the stage
            # call sites spell cache["key"], so the signature must keep one
            # parameter per container, not per projected tensor).
            raise ValueError("pipeline forward_args cannot be combined with container inputs")
        expected_forward = inputs + runtime_scalars + passthrough_names
        if (not isinstance(forward_args, list) or len(forward_args) != len(expected_forward)
                or any(not isinstance(name, str) or name not in expected_forward for name in forward_args)
                or len(set(forward_args)) != len(forward_args)):
            raise ValueError("pipeline forward_args must be a permutation of inputs, runtime "
                             "scalars, and passthrough forward args")
        forward_params = forward_args + ignored
    else:
        if passthrough_names:
            raise ValueError("pipeline passthrough_forward_args requires forward_args to fix positions")
        # forward_params here is the container-built list from above: one
        # parameter per plain input or per container. Appending (instead of
        # rebuilding as inputs + runtime_scalars) keeps plain pipelines
        # byte-identical while container pipelines get the signature their
        # stage call spellings actually reference.
        forward_params = forward_params + ignored + runtime_scalars
    constructor_names = []
    constructor_types = {}
    constructor_args = []
    saw_default = False
    for item in constructor:
        if not isinstance(item, dict):
            raise ValueError("pipeline constructor entries must be objects")
        name = _identifier(item.get("name"), "pipeline constructor argument")
        dtype = item.get("dtype")
        if (name in inputs or name in constructor_types
                or dtype not in ("int32", "int64", "float32", "bool", "ignored")):
            raise ValueError("invalid pipeline constructor argument")
        if "default" in item:
            saw_default = True
        elif saw_default:
            raise ValueError("required pipeline constructor argument follows a default")
        constructor_names.append(name)
        constructor_types[name] = dtype
        constructor_args.append(name + ("=" + repr(item["default"]) if "default" in item else ""))
    available = set(inputs)
    stage_lines = []
    generated = []
    program_hash = hashlib.sha256()
    spec_hash = hashlib.sha256(manifest.read_bytes())
    seen_functions = set()
    seen_stages = set()
    kernels = output_dir / "kernels"
    kernels.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="littlekernel-pipeline-") as temporary:
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ValueError("pipeline stages must be objects")
            name = _identifier(stage.get("name"), "stage name")
            if name in seen_stages:
                raise ValueError("pipeline stage names must be unique")
            seen_stages.add(name)
            source = selected / "stages" / name / "draft.py"
            spec_path = selected / "stages" / name / "export.json"
            if not source.is_file() or not spec_path.is_file():
                raise FileNotFoundError(f"stage {name} needs draft.py and export.json")
            spec = load_spec(spec_path).legacy_mapping()
            function = spec["_function"]
            if function in seen_functions:
                raise ValueError("pipeline stage functions must be unique")
            seen_functions.add(function)
            stage_inputs = [item["name"] for item in spec["inputs"]]
            # A stage-declared tensor_list consumes a pipeline-level tensor
            # list forwarded as one argument (see vibe_cuda_eval tensor_lists).
            stage_inputs += [item["name"] for item in spec["tensor_lists"]]
            if any(item not in available for item in stage_inputs):
                raise ValueError(f"stage {name} uses unavailable inputs")
            for item in spec["constructor"]:
                if constructor_types.get(item["name"]) != item["dtype"]:
                    raise ValueError(f"stage {name} constructor differs from pipeline")
            stage_outputs = [item["name"] for item in spec["_outputs"]]
            # A self-aliased output renames the stage input it updates in
            # place; it must not be treated as a fresh tensor.
            inplace_updates = {item["name"] for item in spec["_outputs"] if item.get("alias_of") == item["name"]}
            if any(item not in stage_inputs for item in inplace_updates):
                raise ValueError(f"stage {name} in-place output must name a stage input")
            fresh_outputs = [item for item in stage_outputs if item not in inplace_updates]
            if any(item in available or item in constructor_names for item in fresh_outputs):
                raise ValueError(f"stage {name} redefines a tensor")
            workspace = Path(temporary) / f"stage-{index:03d}"
            (workspace / "task").mkdir(parents=True)
            agent = workspace / "agent"
            agent.mkdir()
            draft_copy = agent / "draft.py"
            spec_copy = agent / "export.json"
            shutil.copy2(source, draft_copy)
            shutil.copy2(spec_path, spec_copy)
            cuda, binding, _ = export(draft_copy, spec_copy, workspace, force=True)
            for file in (cuda, binding):
                destination = kernels / file.name
                if destination.exists():
                    raise ValueError(f"duplicate generated source: {file.name}")
                shutil.copy2(file, destination)
                generated.append(str(destination.relative_to(output_dir)))
            # Stage call sites substitute container spellings (cache["ck"])
            # for tensor names projected out of a forward container argument.
            args = [arg_spelling.get(item, item) for item in stage_inputs]
            stage_runtime = [item["name"] for item in spec["scalars"] if item.get("_runtime")]
            if any(name not in runtime_scalars for name in stage_runtime):
                raise ValueError(f"stage {name} consumes an undeclared runtime scalar")
            # Forward scalars in the stage's own export.json declaration
            # order: that is the stage's positional C++/binding ABI. The
            # pipeline declaration order is only the Model.forward ABI and
            # must not be imposed on a stage whose binding lists the same
            # scalars in a different order (distinct alpha/beta values would
            # otherwise swap silently).
            args += stage_runtime
            # constexpr constructor entries are compile-time bound and must not
            # be forwarded as runtime args; in-place (self-aliased) outputs are
            # not fresh results and stay bound through their input aliases.
            args += [
                "self." + item["name"]
                for item in spec["constructor"]
                if item["dtype"] != "ignored" and "constexpr" not in item
            ]
            result = ", ".join(fresh_outputs)
            if len(fresh_outputs) > 1:
                result = "(" + result + ")"
            call = f"cuda_extension.agent_{function}({', '.join(args)})"
            if not fresh_outputs:
                # Fully in-place stage: buffers are updated through their
                # aliases, so there is no fresh result to bind and the
                # caller's original objects keep their identity.
                stage_lines.append(f"        {call}")
            elif len(stage_outputs) == len(fresh_outputs):
                stage_lines.append(f"        {result} = {call}")
            else:
                # Mixed fresh and in-place results: index the returned tuple.
                temp = f"lk_stage_{index}_result"
                stage_lines.append(f"        {temp} = {call}")
                for position, item in enumerate(stage_outputs):
                    if item not in inplace_updates:
                        stage_lines.append(f"        {item} = {temp}[{position}]")
            available.update(stage_outputs)
            program_hash.update(source.read_bytes())
            spec_hash.update(spec_path.read_bytes())
    if any(name not in available for name in outputs):
        raise ValueError("pipeline outputs were not produced")
    stored = [name for name in constructor_names if constructor_types[name] != "ignored"]
    if any(name in constructor_names for name in passthrough_names):
        raise ValueError("pipeline passthrough forward arguments must not collide with constructor")
    init = f"    def __init__(self{', ' if constructor_args else ''}{', '.join(constructor_args)}):\n"
    init += "        super().__init__()\n"
    init += "".join(f"        self.{name} = {name}\n" for name in stored)
    result = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
    model = ("# Generated by LittleKernel for ai-compiler-eval.\n"
             "import torch\nimport cuda_extension\n\n\n"
             "class Model(torch.nn.Module):\n" + init + "\n"
             f"    def forward(self, {', '.join(forward_params)}):\n" + "\n".join(stage_lines) + "\n"
             f"        return {result}\n")
    (output_dir / "model_new.py").write_text(model)
    generated.append("model_new.py")
    return {
        "backend": "cuda-extension",
        "compiler": "LittleKernel",
        "frontend": "agent-authored-littlekernel-pipeline",
        "case_index": case_index,
        "program_sha256": program_hash.hexdigest(),
        "export_spec_sha256": spec_hash.hexdigest(),
        "generated": generated,
    }


# Missing programs are an honest failure, never an oracle-backed candidate:
# do not delegate to the task's reference Model, do not emit never-launched
# placeholder CUDA, and write nothing to the output dir for a missing case.
# Present-but-malformed programs still raise their real errors below.


def lower(task_dir: Path, program_dir: Path, output_dir: Path, case_index: int) -> dict:
    if not (task_dir / "model.py").is_file() or not (task_dir / "test_cases.py").is_file():
        raise ValueError("task requires model.py and test_cases.py")
    # A case directory selects the export manifest for this case. When it carries no
    # draft.py, the shared top-level draft.py is specialized by the case manifest
    # (e.g. per-case constexpr constructor values) instead of duplicating source.
    case_dir = program_dir / f"case-{case_index:03d}"
    selected = (case_dir if
                (case_dir / "pipeline.json").is_file() or (case_dir / "export.json").is_file() else program_dir)
    pipeline = selected / "pipeline.json"
    if pipeline.is_file():
        return _pipeline(selected, pipeline, output_dir, case_index)
    spec = selected / "export.json"
    draft = selected / "draft.py"
    if not draft.is_file() and selected != program_dir:
        draft = program_dir / "draft.py"
    if not draft.is_file() or not spec.is_file():
        # Honest unsupported/missing-entry failure (source policy): a case
        # without an authored LittleKernel entry is owner authoring work. It
        # must NOT degrade to executing the task's reference model, an eager
        # Torch oracle, or a no-op CUDA placeholder labelled as compiled.
        raise NotImplementedError("unsupported: no LittleKernel draft.py and export.json for case "
                                  f"{case_index} in {program_dir}")

    with tempfile.TemporaryDirectory(prefix="littlekernel-export-") as temporary:
        workspace = Path(temporary)
        (workspace / "task").mkdir()
        agent = workspace / "agent"
        agent.mkdir()
        source_copy = agent / "draft.py"
        spec_copy = agent / "export.json"
        shutil.copy2(draft, source_copy)
        shutil.copy2(spec, spec_copy)
        cuda, binding, model = export(source_copy, spec_copy, workspace, force=True)
        kernels = output_dir / "kernels"
        kernels.mkdir(parents=True, exist_ok=True)
        generated = []
        for source, destination in (
            (cuda, kernels / cuda.name),
            (binding, kernels / binding.name),
            (model, output_dir / "model_new.py"),
        ):
            shutil.copy2(source, destination)
            generated.append(str(destination.relative_to(output_dir)))
    return {
        "backend": "cuda-extension",
        "compiler": "LittleKernel",
        "frontend": "agent-authored-littlekernel",
        "case_index": case_index,
        "program_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
        "export_spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "generated": generated,
    }
