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
Registry for Enum types used in special structs.

This module provides a mechanism to register Enum classes that are used
in special struct template parameters, so that their C++ enum class
definitions can be generated automatically.
"""

from typing import Dict, Optional
from enum import Enum

# Registry: enum_name -> enum_class
_enum_registry: Dict[str, type] = {}


def register_enum(enum_name=None):
    """
    Decorator to register an Enum class for code generation.
    
    Usage:
        @register_enum
        class GemmType(Enum):
            Normal = 0
        
        # Or with custom name:
        @register_enum("CustomGemmType")
        class GemmType(Enum):
            Normal = 0
    
    Args:
        enum_name: Optional name for the enum. If not provided, uses the class name.
    """
    # Support both @register_enum and @register_enum("name")
    if enum_name is None or isinstance(enum_name, str):
        # Called as @register_enum() or @register_enum("name")
        def decorator(enum_class: type) -> type:
            if not issubclass(enum_class, Enum):
                raise ValueError(f"{enum_class} is not an Enum class")
            name = enum_name if enum_name is not None else enum_class.__name__
            _enum_registry[name] = enum_class
            return enum_class

        return decorator
    else:
        # Called as @register_enum without parentheses
        enum_class = enum_name
        if not issubclass(enum_class, Enum):
            raise ValueError(f"{enum_class} is not an Enum class")
        name = enum_class.__name__
        _enum_registry[name] = enum_class
        return enum_class


def register_enum_legacy(enum_name: str, enum_class: type) -> None:
    """
    Legacy function to register an Enum class for code generation.
    
    Args:
        enum_name: Name of the enum (e.g., 'GemmType')
        enum_class: The Enum class object
    """
    if not issubclass(enum_class, Enum):
        raise ValueError(f"{enum_class} is not an Enum class")
    _enum_registry[enum_name] = enum_class


def get_enum(enum_name: str) -> Optional[type]:
    """Get a registered Enum class by name."""
    return _enum_registry.get(enum_name)


def get_all_registered_enums() -> Dict[str, type]:
    """Get all registered Enum classes."""
    return _enum_registry.copy()


def generate_enum_cpp_definition(enum_name: str, enum_class: type) -> str:
    """
    Generate C++ enum class definition from Python Enum class.
    
    Args:
        enum_name: Name of the enum (e.g., 'GemmType')
        enum_class: The Enum class object
        
    Returns:
        C++ enum class definition as string
    """
    lines = [f"enum class {enum_name} {{"]

    # Get all enum members
    members = list(enum_class)

    # Generate enum members
    for i, member in enumerate(members):
        member_name = member.name
        member_value = member.value
        if i < len(members) - 1:
            lines.append(f"    {member_name} = {member_value},")
        else:
            lines.append(f"    {member_name} = {member_value}")

    lines.append("};")
    return "\n".join(lines)
