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

from little_kernel.language.builtin_base import Builtin, builtin, BuiltinOp
from little_kernel.core.type_system import int32_t

nvshmem_includes = ["<nvshmem.h>", "<nvshmemx.h>"]


@builtin
class my_pe(BuiltinOp):
    """Get the PE (processing element) ID of the calling PE."""

    @staticmethod
    def return_type():
        return int32_t

    @staticmethod
    def codegen():
        return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_my_pe()")


@builtin
class n_pes(BuiltinOp):
    """Get the total number of PEs."""

    @staticmethod
    def return_type():
        return int32_t

    @staticmethod
    def codegen():
        return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_n_pes()")


@builtin
class team_my_pe(BuiltinOp):
    """Get the PE ID within a team."""

    @staticmethod
    def return_type(team):
        return int32_t

    @staticmethod
    def codegen(team):
        return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_team_my_pe({})".format(team))


@builtin
class team_n_pes(BuiltinOp):
    """Get the number of PEs in a team."""

    @staticmethod
    def return_type(team):
        return int32_t

    @staticmethod
    def codegen(team):
        return Builtin(body="", includes=nvshmem_includes, return_val="nvshmem_team_n_pes({})".format(team))


@builtin
class int_p(BuiltinOp):
    """Put an integer value to a remote PE."""

    @staticmethod
    def return_type(dest, value, pe):
        return int32_t

    @staticmethod
    def codegen(dest, value, pe):
        return Builtin(body="", includes=nvshmem_includes,
                       return_val="nvshmem_int_p({}, {}, {})".format(dest, value, pe))


@builtin
class remote_ptr(BuiltinOp):
    """Get a pointer to a symmetric object on a remote PE."""

    @staticmethod
    def return_type(ptr_type, pe_type):
        return ptr_type

    @staticmethod
    def codegen(local_ptr, pe):
        return Builtin(body="", includes=nvshmem_includes,
                       return_val=f"reinterpret_cast<decltype({local_ptr})>(nvshmem_ptr((void*){local_ptr}, {pe}))")


@builtin
class remote_mc_ptr(BuiltinOp):
    """Get a multicast pointer for a team."""

    @staticmethod
    def return_type(ptr_type, team_type):
        return ptr_type

    @staticmethod
    def codegen(team, ptr):
        return Builtin(body="", includes=nvshmem_includes,
                       return_val=f"reinterpret_cast<decltype({ptr})>(nvshmemx_mc_ptr((void*){ptr}, {team}))")
