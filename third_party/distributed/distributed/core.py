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
from triton.language import semantic
import builtins


# adapted from python/triton/language/core.py
def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _builder=None):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :param _builder: the builder
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
        call = func(lib_name, lib_path, symbol, arg_list, [ret_type.to_ir(_builder) for ret_type in ret_types], is_pure)

        if len(ret_types) == 0:
            return tensor(call, tl.void)
        if len(ret_types) == 1:
            return tensor(call.get_result(0), ret_types[0])
        return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))


@builtin
def extern_call(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _builder=None):
    '''
        Dispatch an function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :param _builder: the builder
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = semantic.to_tensor(dispatch_args[i], _builder)
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

    func = _builder.create_extern_call
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, is_pure, _builder)
