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

from triton.language import core as tl
from triton.language.core import builtin, tensor
from typing import List
import builtins


# adapted from python/triton/language/core.py
def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _semantic=None):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :param _semantic: the builder
        :return: the return value of the function
    '''
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_types = arg_type_symbol_dict[arg_types][1]
        if not isinstance(ret_types, (List, tuple)):
            ret_types = [ret_types]

        if symbol == "":
            raise ValueError("Symbol can not be empty")
        call = func(lib_name, lib_path, symbol, arg_list, [ret_type.to_ir(_semantic.builder) for ret_type in ret_types],
                    is_pure)

        if len(ret_types) == 0:
            return tensor(call, tl.void)
        if len(ret_types) == 1:
            return tensor(call.get_result(0), ret_types[0])
        return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


@builtin
def extern_call(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _semantic=None):
    '''
        Dispatch an function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :param _semantic: the semantic
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if not all_scalar:
        raise ValueError("extern call only support inputs with scalr type")

    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    # `distributed.extern_call` needs string attributes (lib/path/symbol) that the
    # OpInfo plugin ABI (Value-only) cannot carry, so it is created via the
    # companion pybind11 module instead of a fork-patched builder method.
    from triton_dist._plugin import load_ext_module
    ext = load_ext_module()
    if ext is None:
        raise RuntimeError("extern_call requires the companion module libtriton_dist_ext; build it "
                           "via plugin/build.sh (TRITON_DIST_BUILD_PY_EXT=ON).")
    builder = _semantic.builder

    def func(lib, path, symbol, arg_list, ret_ir_types, pure):
        return ext.create_extern_call(builder, lib, path, symbol, arg_list, ret_ir_types, pure)

    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, is_pure, _semantic)


# adapted from python/triton/language/core.py: support pointer type inputs
@builtin
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool,
                       _semantic=None, check_args=True):
    '''
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :param _semantic: the semantic
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    arg_types = tuple(arg_types)
    # On Triton 3.7.1, ``tl.dispatch`` takes the *return dtype* (``ret_type``) at
    # this position -- NOT a ``ret_shape`` as the legacy 3.4 fork did. Passing a
    # shape (``None`` for scalar externs) here is what caused
    # ``'NoneType' object has no attribute 'to_ir'`` deep in dispatch. Resolve the
    # return dtype from the (arg-types -> (symbol, ret_type)) table and, for block
    # inputs, re-shape it to the broadcast type -- mirroring upstream
    # ``tl.extern_elementwise``.
    ret_type = arg_type_symbol_dict[arg_types][1]
    if len(arg_types) > 0:
        arithmetic_check = True
        # If there's a type tuple that is not supported by the library, we will do arithmetic check
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        if check_args:
            if arithmetic_check:
                # Get the broadcast shape over all the arguments
                for item in dispatch_args:
                    _, broadcast_arg = _semantic.binary_op_type_checking_impl(item, broadcast_arg, allow_lhs_ptr=True,
                                                                              allow_rhs_ptr=True,
                                                                              arithmetic_check=arithmetic_check)
                # Change the shape of each argument based on the broadcast shape
                for i in builtins.range(len(dispatch_args)):
                    dispatch_args[i], _ = _semantic.binary_op_type_checking_impl(dispatch_args[i], broadcast_arg,
                                                                                 allow_lhs_ptr=True, allow_rhs_ptr=True,
                                                                                 arithmetic_check=arithmetic_check)
            else:
                # Types matched — skip type check, only find broadcast shape.
                for item in dispatch_args:
                    if item.type.is_block():
                        broadcast_arg = item
                        break
            if not all_scalar:
                ret_type = broadcast_arg.type.with_element_ty(ret_type)
    func = _semantic.builder.create_extern_elementwise
    return tl.dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_type, is_pure, _semantic)
