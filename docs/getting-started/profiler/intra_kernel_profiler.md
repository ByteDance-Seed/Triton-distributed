# Intra-Kernel Profiler User Guide
This guide details the function and interface usage of the Intra-Kernel Profiler, which profiles the execution time of each task in the kernel to guide performance optimization.

The profiler provides separate interfaces for **Device Side** and **Host Side** with the following usage:


## 1. Device Side Interfaces
Used to initialize the profiler and record start/end times of target tasks within the kernel.

### 1.0 Dependencies
The intra-kernel profiler relies on `tg4perfetto` to generate trace for now.
Install `tg4perfetto` first:
```bash
pip install tg4perfetto
```

### 1.1 Profiler Initialization
Create a profiler instance with configuration parameters:
```python
from triton_dist.tools.profiler import Profiler

profiler = Profiler.create(
    profiler_buffer=profiler_buf,
    group_id=0,
    num_groups=1,
    is_leader=(tid(0) == 0),
    ENABLE_PROFILING=True
)
```

- `profiler_buffer`: Device side tensor passed from the host side to the kernel.
- `group_id`: For the current Triton frontend, **set to 0**.
- `num_groups`: Total number of thread groups in a block. For the triton, **set to 1** is enough.
- `is_leader`: Predicate to select one thread per group (e.g., tid(0) == 0) to perform record.
- `ENABLE_PROFILING`: Default to true, if set to False to skip all record operations.

### 1.2 Task Time Record
Record start and end times of target tasks using the record method:

```python
# Record task start
profiler = profiler.record(is_start=True, task_type=0)
# do something.... 
# Record task end
profiler = profiler.record(is_start=False, task_type=0)
```

- `is_start`: Distinguish between the task start and end, True (start) / False (end).
- `task_type`: Integers start from 0 will be mapped to the corresponding task name during visualization(e.g. task_type=0 -> "perfect")

> Note: The Triton frontend does not support in-place modification. Thus, `profiler.record` returns a new profiler instance that overwrites the original.

## 2. Host Side Interfaces
Used to manage profiler buffers and export trace files, with two usage modes: Wrapped Interface (simplified) and Separate Interfaces (flexible).

### 2.1 Wrapped Interface: ProfilerBuffer
A context-manager-based interface simplifying buffer management and trace export:

```python
from triton_dist.tools.profiler import ProfilerBuffer

with ProfilerBuffer(
    max_num_profile_slots=1000000,
    trace_file="copy",
    task_names=["perfect", "non-perfect"]
) as profile_buf:
    # Execute Triton kernel (pass profile_buf as parameter)
    copy_1d_tilewise_kernel[grid](
        profile_buf, src_tensor, dst_tensor, grid_barrier, M * N
    )
```

- `max_num_profile_slots`: Must be greater than the total number of record operations across all thread blocks (user responsibility).
- `trace_file`	Output trace file name.
- `task_names`	List of readable names corresponding to task_type (e.g., task_type=0 -> "perfect").

By default, a trace file is generated for each iteration. To export traces selectively, use these switches:

```python
from triton_dist.tools.profiler import set_export_trace_on, set_export_trace_off

set_export_trace_on()  # Enable export on ProfilerBuffer exit

set_export_trace_off()  # Disable export
```

### 2.2 Separate Interfaces
For fine-grained control, use independent functions for buffer management and trace export:
```python
from triton_dist.tools.profiler import (
    alloc_profiler_buffer, 
    reset_profiler_buffer,
    export_to_perfetto_trace
)

# Allocate profiler buffer
profile_buf = alloc_profiler_buffer(max_num_profile_slots=1000000)

# Reset buffer
reset_profiler_buffer(profile_buf)

# Execute Triton kernel
copy_1d_tilewise_kernel[grid](
    profile_buf, src_tensor, dst_tensor, grid_barrier, M * N
)

# Export trace data
export_to_perfetto_trace(
    profiler_buffer=profile_buf,
    task_names=["perfect", "non-perfect"],
    file_name="copy"
)
```

## 3. Reference
- [feat: flashinfer intra-kernel profiler](https://github.com/flashinfer-ai/flashinfer/pull/913)
