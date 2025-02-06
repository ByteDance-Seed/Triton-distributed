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
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>
#include <torch/python.h>

namespace py = pybind11;

namespace distributed {

// Registry of functions that register
// functions into module
class OpInitRegistry {
public:
  using OpInitFunc = std::function<void(py::module &)>;
  static OpInitRegistry &instance();
  void register_one(std::string name, OpInitFunc &&func);
  void initialize_all(py::module &m) const;

private:
  std::map<std::string, OpInitFunc> registry_;
  mutable std::mutex register_mutex_;

  OpInitRegistry() {}
  OpInitRegistry(const OpInitRegistry &) = delete;
  OpInitRegistry &operator=(const OpInitRegistry &) = delete;
};

template <typename T>
struct TorchClassWrapper : public torch::CustomClassHolder, T {
public:
  using T::T;
};

} // namespace distributed