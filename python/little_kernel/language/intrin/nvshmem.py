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

from ..builtin_base import Builtin, builtin
from little_kernel.core.type_system import int32_t

nvshmem_includes = ["<nvshmem.h>", "<nvshmemx.h>"]


def codegen_my_pe():
    return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_my_pe()")


@builtin(eval_return_type=int32_t, codegen_func=codegen_my_pe)
def my_pe():
    raise RuntimeError("my_pe should never be called in compilation")


def codegen_n_pes():
    return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_n_pes()")


@builtin(eval_return_type=int32_t, codegen_func=codegen_n_pes)
def n_pes():
    raise RuntimeError("n_pes should never be called in compilation")


def codegen_team_my_pe(team):
    return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_team_my_pe({})".format(team))


@builtin(eval_return_type=int32_t, codegen_func=codegen_team_my_pe)
def team_my_pe(team):
    raise RuntimeError("team_my_pe should never be called in compilation")


def codegen_team_n_pes(team):
    return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_team_n_pes({})".format(team))


@builtin(eval_return_type=int32_t, codegen_func=codegen_team_n_pes)
def team_n_pes(team):
    raise RuntimeError("team_n_pes should never be called in compilation")


def codegen_int_p(dest, value, pe):
    return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_int_p({}, {}, {})".format(dest, value, pe))


@builtin(eval_return_type=int32_t, codegen_func=codegen_int_p)
def int_p(dest, value, pe):
    raise RuntimeError("int_p should never be called in compilation")


def codegen_remote_ptr(local_ptr, pe):
    return Builtin(body="", includes=nvshmem_includes,
                   return_val=f"reinterpret_cast<decltype({local_ptr})>(nvshmem_ptr((void*){local_ptr}, {pe}))")


@builtin(eval_return_type=lambda ptr_type, pe_type: ptr_type, codegen_func=codegen_remote_ptr)
def remote_ptr(local_ptr, pe):
    raise RuntimeError("remote_ptr should never be called in compilation")


def codegen_remote_mc_ptr(team, ptr):
    return Builtin(body="", includes=nvshmem_includes,
                   return_val=f"reinterpret_cast<decltype({ptr})>(nvshmemx_mc_ptr((void*){ptr}, {team}))")


@builtin(eval_return_type=lambda ptr_type, team_type: ptr_type, codegen_func=codegen_remote_mc_ptr)
def remote_mc_ptr(team, ptr):
    raise RuntimeError("remote_mc_ptr should never be called in compilation")
