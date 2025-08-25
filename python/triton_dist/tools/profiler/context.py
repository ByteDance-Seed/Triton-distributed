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

PROFILER_ENTRY_DTYPE = torch.uint64
EMPTY_VALUE = 0xFFFFFFFFFFFFFFFF
# Whether to export trace. Default is True.
export_trace_on = True


def set_export_trace_on():
    global export_trace_on
    export_trace_on = True


def set_export_trace_off():
    global export_trace_on
    export_trace_on = False


def get_export_trace_on():
    global export_trace_on
    return export_trace_on


def alloc_profiler_buffer(max_num_profile_slots):
    buf = torch.empty((max_num_profile_slots, ), dtype=PROFILER_ENTRY_DTYPE, device=torch.cuda.current_device())
    buf = reset_profiler_buffer(buf)
    return buf


def reset_profiler_buffer(buf):
    buf.fill_(EMPTY_VALUE)
    return buf


def is_empty_slot(val):
    return val == EMPTY_VALUE


class ProfilerBuffer:

    def __init__(self, max_num_profile_slots, trace_file, task_names):
        self.trace_file = trace_file
        self.task_names = task_names
        self.profiler_buffer = alloc_profiler_buffer(max_num_profile_slots)

    def __enter__(self):
        reset_profiler_buffer(self.profiler_buffer)
        return self.profiler_buffer

    def __exit__(self, *args, **kwargs):
        if get_export_trace_on():
            from .viewer import export_to_perfetto_trace
            export_to_perfetto_trace(self.profiler_buffer, self.task_names, self.trace_file, verbose=False)
