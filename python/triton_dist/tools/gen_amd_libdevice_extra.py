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
import argparse
import sys
from contextlib import redirect_stdout

ALIGNMENT_MAP = {"i32": 4, "i64": 8}

LD_SEMANTIC = ["acquire", "monotonic"]
ST_SEMANTIC = ["monotonic", "release"]
FENCE_SEMANTIC = ["acquire", "release", "acq_rel", "seq_cst"]
ALL_SEMANTIC = ["acquire", "monotonic", "release", "acq_rel"]
# LLVM IR has no unsigned data type. so only signed here
DTYPES = ["i32", "i64"]
SCOPES = ["workgroup", "agent", "system"]

# some references:
# https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/LangRef.html#load-instruction
# https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/LangRef.html#store-instruction
# https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/LangRef.html#i-atomicrmw
# https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/LangRef.html#cmpxchg-instruction
# A sample for load/store
"""
define linkonce hidden i32 @__load_acquire_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("workgroup-one-as") acquire, align 4
  ret i32 %2
}

define linkonce hidden i32 @__store_relaxed_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("workgroup-one-as") relaxed, align 4
  ret i32 %1
}

define linkonce hidden i32 @__atomic_add_release_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("agent-one-as") release, align 4
  ret i32 %3
}

define linkonce hidden i32 @__atom_cas_acquire_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("agent-one-as") acquire monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}
"""

LOAD_TEMPLATE = """
define linkonce hidden {dtype} @__load_{semantic}_{scope}_{dtype}(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {{
  %2 = load atomic {dtype}, ptr addrspace(1) %0 {scope_attr} {semantic}, align {alignment}
  ret {dtype} %2
}}
"""
STORE_TEMPLATE = """
define linkonce hidden {dtype} @__store_{semantic}_{scope}_{dtype}(ptr addrspace(1) inreg readonly captures(none) %0, {dtype} %1) local_unnamed_addr {{
  store atomic {dtype} %1, ptr addrspace(1) %0 {scope_attr} {semantic}, align {alignment}
  ret {dtype} %1
}}
"""
ATOM_ADD_TEMPLATE = """
define linkonce hidden {dtype} @__atomic_add_{semantic}_{scope}_{dtype}(ptr addrspace(1) inreg readonly captures(none) %0, {dtype} %1) local_unnamed_addr {{
  %3 = atomicrmw add ptr addrspace(1) %0, {dtype} %1 {scope_attr} {semantic}, align {alignment}
  ret {dtype} %3
}}
"""
ATOM_CAS_TEMPLATE = """
define linkonce hidden {dtype} @__atom_cas_{success_semantic}_{failure_semantic}_{scope}_{dtype}(ptr addrspace(1) inreg readonly captures(none) %0, {dtype} %1, {dtype} %2) local_unnamed_addr {{
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, {dtype} %1, {dtype} %2 {scope_attr} {success_semantic} {failure_semantic}, align {alignment}
  %11 = extractvalue {{ {dtype}, i1 }} %10, 0
  ret {dtype} %11
}}
"""
FENCE_TEMPLACE = """define linkonce hidden void @__extra_fence_{semantic}_{scope}() local_unnamed_addr {{
  fence syncscope("{scope}") {semantic}
  ret void
}}
"""


def gen_load():
    for dtype in DTYPES:
        for semantic in LD_SEMANTIC:
            for scope in SCOPES:
                scope_attr = "" if scope == "system" else f'syncscope("{scope}-one-as")'
                print(
                    LOAD_TEMPLATE.format(
                        semantic=semantic,
                        scope=scope,
                        scope_attr=scope_attr,
                        dtype=dtype,
                        alignment=ALIGNMENT_MAP[dtype],
                    ))


def gen_store():
    for dtype in DTYPES:
        for semantic in ST_SEMANTIC:
            for scope in SCOPES:
                scope_attr = "" if scope == "system" else f'syncscope("{scope}-one-as")'
                print(
                    STORE_TEMPLATE.format(
                        semantic=semantic,
                        scope=scope,
                        scope_attr=scope_attr,
                        dtype=dtype,
                        alignment=ALIGNMENT_MAP[dtype],
                    ))


def gen_atomic_add():
    for dtype in DTYPES:
        for semantic in ALL_SEMANTIC:
            for scope in SCOPES:
                scope_attr = "" if scope == "system" else f'syncscope("{scope}-one-as")'
                print(
                    ATOM_ADD_TEMPLATE.format(
                        semantic=semantic,
                        scope=scope,
                        scope_attr=scope_attr,
                        dtype=dtype,
                        alignment=ALIGNMENT_MAP[dtype],
                    ))


def gen_atomic_cas():
    for dtype in DTYPES:
        for semantic in ALL_SEMANTIC:
            for scope in SCOPES:
                scope_attr = "" if scope == "system" else f'syncscope("{scope}-one-as")'
                print(
                    ATOM_CAS_TEMPLATE.format(
                        success_semantic=semantic,
                        failure_semantic="monotonic",
                        scope=scope,
                        scope_attr=scope_attr,
                        dtype=dtype,
                        alignment=ALIGNMENT_MAP[dtype],
                    ))


def gen_fence():
    for semantic in FENCE_SEMANTIC:
        for scope in SCOPES:
            print(FENCE_TEMPLACE.format(
                semantic=semantic,
                scope=scope,
            ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", action="store_true", default=False, help="file path to save llvm ir to")
    return parser.parse_args()


args = parse_args()

with redirect_stdout(open(args.out) if args.out else sys.stdout):
    gen_load()
    gen_store()
    gen_atomic_add()
    gen_atomic_cas()
    gen_fence()

# print("done")
