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
"""
Pass configuration and registration.

This module handles the registration of compilation passes for different backends.
Passes can be registered here or imported from other modules.
"""

import os
from typing import List, Callable
from .pass_registry import register_pass, get_passes, empty_pass
from .dump import dump_ast

# Environment variables
PASS_DEBUG = os.getenv("PASS_DEBUG", "").split(",")
if "" in PASS_DEBUG:
    PASS_DEBUG.remove("")
PASS_DEBUG_ALL = "all" in PASS_DEBUG
SIMPLE_MODE = os.getenv("SIMPLE_MODE", "true") == "true"


def register_default_passes(backend: str = "cuda"):
    """
    Register default passes for a backend.
    
    Args:
        backend: Backend name (e.g., "cuda")
    """
    from .special_struct_materialize_pass import special_struct_materialize_pass
    from .method_to_intrin import method_to_intrin
    from .constfold import const_fold
    from .inline import inline
    from .flatten_empty import flatten_empty
    from .insert_mem_alloc import insert_mem_alloc

    # Register initial dump pass if debug is enabled
    if len(PASS_DEBUG) > 0:
        register_pass(backend, dump_ast("initial"), "initial", enable_debug=False)
    else:
        register_pass(backend, empty_pass, "empty", enable_debug=False)

    # Register other passes
    register_pass(backend, special_struct_materialize_pass, "special_struct_materialize_pass", simple_mode=SIMPLE_MODE)
    register_pass(backend, method_to_intrin, "method_to_intrin", simple_mode=SIMPLE_MODE)
    register_pass(backend, const_fold, "const_fold", simple_mode=SIMPLE_MODE)
    register_pass(backend, inline, "inline", simple_mode=SIMPLE_MODE)
    register_pass(backend, flatten_empty, "flatten_empty", simple_mode=SIMPLE_MODE)
    register_pass(backend, insert_mem_alloc, "insert_mem_alloc", simple_mode=SIMPLE_MODE)


def get_passes_for_backend(backend: str = "cuda") -> List[Callable]:
    """
    Get the list of passes for a backend.
    
    Args:
        backend: Backend name
        
    Returns:
        List of pass functions
    """
    return get_passes(backend)


# Auto-register default passes for CUDA backend
register_default_passes("cuda")
