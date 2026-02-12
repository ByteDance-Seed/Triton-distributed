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

# Import pass_registry first (no dependencies)
from .pass_registry import (  # noqa: F401
    get_pass_registry, register_pass, get_passes, empty_pass, debug_wrapper,
)

# Define PASSES as a placeholder first to break circular import
# It will be properly initialized after all imports
PASSES = {"cuda": []}  # Placeholder

# Import passes  # noqa: E402
from .pass_base import *  # noqa: F403, E402
from .constfold import *  # noqa: F403, E402
from .inline import *  # noqa: F403, E402
from .mem_analysis import *  # noqa: F403, E402
from .utils.add_parent_reference import *  # noqa: F403, E402
from .insert_mem_alloc import *  # noqa: F403, E402
from .special_struct_materialize_pass import special_struct_materialize_pass as special_struct_materialize_pass  # noqa: E402, F401
from .method_to_intrin import method_to_intrin as method_to_intrin  # noqa: E402, F401
from .flatten_empty import flatten_empty as flatten_empty  # noqa: E402, F401
from .dump import dump_ast as dump_ast  # noqa: E402, F401

# Import pass_config after passes
from .pass_config import (  # noqa: E402, F401
    register_default_passes as register_default_passes, get_passes_for_backend, PASS_DEBUG as PASS_DEBUG, PASS_DEBUG_ALL
    as PASS_DEBUG_ALL, SIMPLE_MODE as SIMPLE_MODE,
)

# Initialize PASSES properly after all imports
PASSES = {"cuda": get_passes_for_backend("cuda")}
