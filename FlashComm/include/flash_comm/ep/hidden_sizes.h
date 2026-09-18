/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

// Single source of truth for explicitly instantiated EP hidden sizes. Add one
// OP(...) line before the terminator to extend both C++ dispatch and generated
// CUDA translation units; setup.py gives those units to Ninja in parallel.
#define FLASH_COMM_SUPPORTED_HIDDEN_SIZES(OP, ...)                             \
  OP(1024, __VA_ARGS__)                                                        \
  OP(1536, __VA_ARGS__)                                                        \
  OP(1792, __VA_ARGS__)                                                        \
  OP(2048, __VA_ARGS__)                                                        \
  OP(2304, __VA_ARGS__)                                                        \
  OP(2432, __VA_ARGS__)                                                        \
  OP(2560, __VA_ARGS__)                                                        \
  OP(2816, __VA_ARGS__)                                                        \
  OP(3072, __VA_ARGS__)                                                        \
  OP(3328, __VA_ARGS__)                                                        \
  OP(3584, __VA_ARGS__)                                                        \
  OP(3840, __VA_ARGS__)                                                        \
  OP(4096, __VA_ARGS__)                                                        \
  OP(4352, __VA_ARGS__)                                                        \
  OP(4864, __VA_ARGS__)                                                        \
  OP(5120, __VA_ARGS__)                                                        \
  OP(6144, __VA_ARGS__)                                                        \
  OP(7168, __VA_ARGS__)                                                        \
  OP(8192, __VA_ARGS__)                                                        \
  OP(12288, __VA_ARGS__)                                                       \
  /* X-macro terminator: keep this line last. */
