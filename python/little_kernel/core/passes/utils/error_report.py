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
Error reporting utilities for compiler passes.

This module provides functions to format error messages with:
- Line and column numbers from AST nodes
- Code snippets showing the problematic code
- Contextual information to help users locate errors
"""

import ast
import inspect
from typing import Optional, Dict, Any, Tuple
import sys


def get_source_lines(func_or_source: Any) -> Optional[list]:
    """
    Get source lines from a function or source code string.
    
    Args:
        func_or_source: Either a function object or a string containing source code
        
    Returns:
        List of source lines, or None if source cannot be retrieved
    """
    if isinstance(func_or_source, str):
        return func_or_source.splitlines(keepends=True)
    elif callable(func_or_source):
        try:
            source = inspect.getsource(func_or_source)
            return source.splitlines(keepends=True)
        except (OSError, TypeError):
            return None
    return None


def get_node_location(node: ast.AST) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Get location information from an AST node.
    
    Args:
        node: AST node
        
    Returns:
        Tuple of (lineno, col_offset, end_lineno, end_col_offset)
        Returns None for missing values
    """
    lineno = getattr(node, 'lineno', None)
    col_offset = getattr(node, 'col_offset', None)
    end_lineno = getattr(node, 'end_lineno', None)
    end_col_offset = getattr(node, 'end_col_offset', None)
    return lineno, col_offset, end_lineno, end_col_offset


def extract_code_snippet(source_lines: list, lineno: int, col_offset: int, end_lineno: Optional[int] = None,
                         end_col_offset: Optional[int] = None, context_lines: int = 10) -> str:
    """
    Extract a code snippet around the specified location.
    
    Args:
        source_lines: List of source code lines
        lineno: Starting line number (1-based)
        col_offset: Starting column offset (0-based)
        end_lineno: Ending line number (1-based, optional)
        end_col_offset: Ending column offset (0-based, optional)
        context_lines: Number of context lines to include before and after
        
    Returns:
        Formatted code snippet string
    """
    if not source_lines or lineno is None:
        return ""

    # Convert to 0-based indexing
    line_idx = lineno - 1
    if line_idx < 0 or line_idx >= len(source_lines):
        return ""

    # Determine range of lines to show
    start_line = max(0, line_idx - context_lines)
    end_line = min(len(source_lines), line_idx + context_lines + 1)

    # If we have end_lineno, extend the range
    if end_lineno is not None:
        end_line_idx = end_lineno - 1
        if end_line_idx >= 0 and end_line_idx < len(source_lines):
            end_line = min(len(source_lines), end_line_idx + context_lines + 1)

    # Build the snippet
    lines = []
    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = source_lines[i].rstrip('\n\r')

        # Mark the problematic line
        if i == line_idx:
            # Add pointer to the column
            if col_offset is not None and col_offset >= 0:
                pointer = " " * col_offset + "^"
                lines.append(f"{line_num:4d} | {line_content}")
                lines.append(f"     | {pointer}")
            else:
                lines.append(f"{line_num:4d} | {line_content}")
        else:
            lines.append(f"{line_num:4d} | {line_content}")

    return "\n".join(lines)


def get_source_from_ctx(ctx: Optional[Dict[str, Any]] = None) -> Optional[list]:
    """
    Try to get source code from context (e.g., from LLKernel).
    
    Args:
        ctx: Context dictionary that may contain source information
        
    Returns:
        List of source lines, or None if not found
    """
    if ctx is None:
        return None

    # Try to get source from __LITTLE_KERNEL_ENTRY__ or other kernel functions
    try:
        from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
        if __LITTLE_KERNEL_ENTRY__ in ctx:
            func = ctx[__LITTLE_KERNEL_ENTRY__]
            return get_source_lines(func)
    except ImportError:
        # Fallback: try common names
        for key in ["__LITTLE_KERNEL_ENTRY__", "__entry__"]:
            if key in ctx:
                func = ctx[key]
                return get_source_lines(func)

    # Try other callable objects in context
    for key, value in ctx.items():
        if callable(value) and hasattr(value, '__code__'):
            source_lines = get_source_lines(value)
            if source_lines:
                return source_lines

    return None


def format_error_message(message: str, node: Optional[ast.AST] = None, source_lines: Optional[list] = None,
                         ctx: Optional[Dict[str, Any]] = None, context_lines: int = 10) -> str:
    """
    Format an error message with location information and code snippet.
    
    Args:
        message: Error message
        node: AST node where the error occurred
        source_lines: Source code lines (optional, will try to infer if not provided)
        context_lines: Number of context lines to show
        
    Returns:
        Formatted error message string
    """
    parts = [message]

    if node is None:
        return message

    # Get location information
    lineno, col_offset, end_lineno, end_col_offset = get_node_location(node)

    # Add location information
    if lineno is not None:
        location_str = f"Line {lineno}"
        if col_offset is not None:
            location_str += f", column {col_offset}"
        if end_lineno is not None and end_lineno != lineno:
            location_str += f" to line {end_lineno}"
        parts.append(f"\nLocation: {location_str}")

    # Try to get code snippet
    if source_lines is None and ctx is not None:
        source_lines = get_source_from_ctx(ctx)

    if source_lines is None:
        # Try to get source from the current frame
        try:
            frame = sys._getframe(2)  # Go up 2 frames to get caller's frame
            # Look for source in frame locals/globals
            for key, value in frame.f_locals.items():
                if callable(value):
                    source_lines = get_source_lines(value)
                    if source_lines:
                        break
            if not source_lines:
                for key, value in frame.f_globals.items():
                    if callable(value):
                        source_lines = get_source_lines(value)
                        if source_lines:
                            break
        except (ValueError, AttributeError):
            pass

    if source_lines and lineno is not None:
        snippet = extract_code_snippet(source_lines, lineno, col_offset, end_lineno, end_col_offset, context_lines)
        if snippet:
            parts.append(f"\nCode snippet:\n{snippet}")

    # Add AST dump for debugging (truncated if too long)
    try:
        ast_dump = ast.dump(node, indent=2)
        if len(ast_dump) > 500:
            ast_dump = ast_dump[:500] + "\n... (truncated)"
        parts.append(f"\nAST node:\n{ast_dump}")
    except Exception:
        pass

    return "\n".join(parts)


class PassError(RuntimeError):
    """
    Base exception class for pass errors with enhanced error reporting.
    """

    def __init__(self, message: str, node: Optional[ast.AST] = None, source_lines: Optional[list] = None,
                 ctx: Optional[Dict[str, Any]] = None, context_lines: int = 10):
        """
        Initialize a pass error.
        
        Args:
            message: Error message
            node: AST node where the error occurred
            source_lines: Source code lines (optional)
            ctx: Context dictionary (optional, used to find source)
            context_lines: Number of context lines to show
        """
        self.node = node
        self.source_lines = source_lines
        formatted_message = format_error_message(message, node, source_lines, ctx, context_lines)
        super().__init__(formatted_message)

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def raise_pass_error(message: str, node: Optional[ast.AST] = None, source_lines: Optional[list] = None,
                     ctx: Optional[Dict[str,
                                        Any]] = None, context_lines: int = 10, error_class: type = PassError) -> None:
    """
    Raise a pass error with formatted message.
    
    Args:
        message: Error message
        node: AST node where the error occurred
        source_lines: Source code lines (optional)
        ctx: Context dictionary (optional, used to find source)
        context_lines: Number of context lines to show
        error_class: Exception class to raise (default: PassError)
        
    Raises:
        error_class: The specified error class with formatted message
    """
    raise error_class(message, node, source_lines, ctx, context_lines)
