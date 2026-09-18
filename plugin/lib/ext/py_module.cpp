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

/// Companion Python extension for the Triton-distributed plugin.
///
/// Why a companion module instead of plain OpInfo callbacks? Upstream Triton
/// 3.7.1's plugin-op binding (python/src/ir.cc) is
/// ``[op](TritonOpBuilder &self, std::vector<Value> args){ op.addOp(self,
/// args); }``
/// -- it passes ``args`` *by value*, inserts no result slot, and returns
/// nothing. So an OpInfo-bound op on 3.7.1 cannot (a) hand a created Value back
/// to Python, nor (b) receive operands under the ``operands[0]=result slot``
/// convention the builders use. (Upstream 3.8 added both via the "[triton-ext]
/// Insert return Value into args for TritonOpBuilder" commit, which 3.7.1
/// predates.) OpInfo also only marshals Values, so it can't carry the string
/// attributes
/// ``distributed.extern_call`` needs, nor return the op/block handles the SIMT
/// region needs. We therefore expose ALL distributed ops here, mirroring the
/// triton-ext/utlx companion-module pattern.
///
/// This is a *separate* pybind11 module (not the dlopen'd plugin .so). Triton
/// 3.7.1 binds its IR with **pybind11** and registers the IR wrapper types
/// (`builder`/`value`/`block`/`OpState`/`type`) with ``py::module_local()`` --
/// i.e. the registrations are private to libtriton's module. pybind11 still
/// lets us:
///   * ACCEPT those types as arguments (e.g. `TritonOpBuilder &`, `Value`): the
///     incoming Python objects carry libtriton's own module-local registration,
///     which pybind11 finds via `try_load_foreign_module_local`. No
///     registration is needed on this side for input-only types.
///   * RETURN those types: pybind11 needs a local registration to build the
///     Python object, so we re-declare `value`/`block`/`OpState`/`type` here as
///     ``py::module_local()`` and mirror (only) the few methods the frontend
///     calls. Objects returned this way are still accepted back by libtriton's
///     builder methods through the same foreign-module-local path. (The
///     companion MUST be built against the SAME pybind11 as libtriton so
///     PYBIND11_INTERNALS_ID matches and the foreign handshake works.)
/// This is what keeps the companion working without patching Triton's ir.cc the
/// way the legacy 3.4 fork did.
///
/// This translation unit (the pybind layer) includes NO Distributed dialect
/// header -- it only forwards to the builder declarations below. The builders
/// themselves (op_builders.cpp / extern_call_builder.cpp /
/// simt_region_builder.cpp, linked into THIS module) do include the dialect
/// headers and call op::create, but those headers only DECLARE each op's
/// `TypeID`; the single TypeID DEFINE and the op/dialect registration live in
/// the dialect TUs compiled into the plugin (libtriton_dist.so). At runtime
/// this module resolves those symbols from the RTLD_GLOBAL plugin, so there is
/// exactly one TypeID owner -- no cross-.so duplication (the
/// load-order-dependent "<op> created with unregistered dialect"). No Triton
/// source edit needed.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/Attributes.h"  // mlir::Attribute
#include "mlir/IR/Block.h"       // mlir::Block / mlir::BlockArgument
#include "mlir/IR/Location.h"    // mlir::Location
#include "mlir/IR/MLIRContext.h" // mlir::MLIRContext
#include "mlir/IR/Operation.h"   // mlir::Operation
#include "python/src/ir.h"       // TritonOpBuilder (registered by libtriton)

#include "extern_call_builder.h" // triton::dist::createExternCallOp
#include "op_builders.h" // triton::dist::ops::create* (value/side-effect ops)
#include "simt_region_builder.h" // triton::dist::create{SimtExecRegion,BlockYield}Op

namespace py = pybind11;
using namespace mlir;

// The distributed op builders (op_builders.cpp, linked into this module) follow
// upstream Triton's plugin-op convention: `operands[0]` is a writable result
// slot, `operands[1..]` are the python-passed inputs. Upstream 3.8 realises
// that convention in `ir.cc` (commit "[triton-ext] Insert return Value into
// args ...") by inserting the slot and returning `args[0]`; **3.7.1 does
// neither** -- its plugin-op binding is `[op](self, args){ op.addOp(self,
// args); }`, passing `args` by value with no slot and no return. So on 3.7.1 an
// OpInfo-bound op both misaligns operands (no slot) and cannot hand a result
// back to Python.
//
// We therefore drive these ops from this companion module instead of OpInfo:
// each wrapper prepends the result slot, calls the builder, and returns
// `args[0]`
// -- locally re-applying the 3.8 convention without any edit to Triton.
// Side-effect ops use the same slot prepend but return nothing.
//
// Operands arrive as a Python list of libtriton-registered `ir.value` (or its
// subclass `ir.block_argument`). pybind11's foreign-module-local loading only
// matches the *exact* C++ type, so a `block_argument` (kernel arg /
// loop-carried value) won't load directly as `Value`. We load it as its exact
// type and upcast (BlockArgument has the same layout as Value). See asValue()
// below.
static Value asValue(const py::handle &h) {
  try {
    return py::cast<Value>(h);
  } catch (const py::cast_error &) {
    return Value(py::cast<BlockArgument>(h));
  }
}
static std::vector<Value> asValues(const py::sequence &seq) {
  std::vector<Value> out;
  out.reserve(py::len(seq));
  for (auto item : seq)
    out.push_back(asValue(item));
  return out;
}

// Value-op wrapper: prepend the result slot, invoke the builder, return the
// populated slot. The builders leave the slot null (and take no action) when
// their operand contract is violated (wrong count, or an attribute operand that
// is not a compile-time constant). Surface that as a clear Python error here
// rather than handing back a null Value that would crash deep inside MLIR on
// first use.
#define DIST_VALUE_OP(pyname, fn)                                              \
  m.def(pyname, [](TritonOpBuilder &self, const py::sequence &ops) -> Value {  \
    std::vector<Value> args = asValues(ops);                                   \
    args.insert(args.begin(), Value());                                        \
    ::triton::dist::ops::fn(self, args);                                       \
    if (!args[0])                                                              \
      throw py::value_error(                                                   \
          std::string(pyname) +                                                \
          ": builder produced no result (malformed operands: wrong operand "   \
          "count or a non-constant attribute operand)");                       \
    return args[0];                                                            \
  })
#define DIST_VOID_OP(pyname, fn)                                               \
  m.def(pyname, [](TritonOpBuilder &self, const py::sequence &ops) {           \
    std::vector<Value> args = asValues(ops);                                   \
    args.insert(args.begin(), Value());                                        \
    ::triton::dist::ops::fn(self, args);                                       \
  })

PYBIND11_MODULE(libtriton_dist_ext, m) {
  m.doc() =
      "Triton-distributed companion builder methods. On Triton 3.7.1 the "
      "plugin-op (OpInfo) binding neither inserts a result slot nor returns "
      "a value, so every distributed op is driven from here instead of "
      "OpInfo: extern_call, simt_exec_region, and the value/side-effect "
      "builder ops (delegating to the plugin's create* functions).";

  // Local re-declarations of libtriton's module-local IR wrapper types so this
  // module can construct (return) them. Method bodies mirror python/src/ir.cc;
  // only the members the SIMT/extern_call/distributed frontend actually invokes
  // are exposed. A value returned from here is a *companion-local* `value`
  // object; method calls dispatch to these bodies (not libtriton's), so the few
  // the frontend uses must be mirrored. Passing it back into a libtriton
  // builder method still works via pybind11 foreign-module-local loading.
  py::class_<Type>(m, "type", py::module_local());
  // Return/param types for the mirrored `value` methods below (get_context /
  // get_loc / set_loc). libtriton registers these `py::module_local()` too, so
  // we must re-declare them locally to be able to build the Python objects on
  // this side; libtriton still accepts them back via the foreign-module-local
  // path.
  py::class_<MLIRContext>(m, "context", py::module_local());
  py::class_<Location>(m, "location", py::module_local());
  // A value returned from this module is a *companion-local* `value`, so EVERY
  // method the Triton frontend invokes on an op-result handle must be mirrored
  // here (dispatch goes to these bodies, not libtriton's) or it surfaces as
  // `AttributeError: 'libtriton_dist_ext.value' object has no attribute ...`.
  // In particular tl.multiple_of / tl.max_contiguous call handle.set_attr +
  // handle.get_context on distributed-op results (e.g. dl.consume_token).
  // Bodies mirror python/src/ir.cc's `value` class.
  py::class_<Value>(m, "value", py::module_local())
      .def("get_type", &Value::getType)
      .def("set_attr",
           [](Value &self, std::string &name, Attribute &attr) -> void {
             // Distributed op results are always OpResults (have a defining
             // op); the block-argument branch is defensive (kernel args are
             // libtriton values and never reach this companion-local type).
             if (Operation *definingOp = self.getDefiningOp()) {
               definingOp->setAttr(name, attr);
             } else if (auto arg = mlir::dyn_cast<BlockArgument>(self)) {
               std::string attrName =
                   name + "_arg" + std::to_string(arg.getArgNumber());
               Block *owner = arg.getOwner();
               if (owner->isEntryBlock())
                 owner->getParentOp()->setAttr(attrName, attr);
             }
           })
      .def("get_context", &Value::getContext)
      .def("get_loc", &Value::getLoc)
      .def("set_loc", &Value::setLoc)
      .def("replace_all_uses_with",
           [](Value &self, Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("id", [](Value &self) { return (uint64_t)self.getImpl(); });
  // BlockArgument is a Value subclass. Register it so a libtriton block
  // argument (e.g. a kernel arg or loop-carried value) passed as an operand can
  // be loaded as `Value` here: pybind11's foreign-module-local fallback only
  // matches the exact C++ type, so without this the base relationship is
  // invisible.
  py::class_<BlockArgument, Value>(m, "block_argument", py::module_local());
  py::class_<Block>(m, "block", py::module_local())
      .def("arg", [](Block &self, int index) -> Value {
        if (index >= static_cast<int>(self.getNumArguments()))
          throw py::index_error("Block argument index out of range");
        return self.getArgument(index);
      });
  py::class_<OpState>(m, "OpState", py::module_local())
      .def("get_num_results",
           [](OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result", [](OpState &self, unsigned idx) -> Value {
        if (idx >= self->getNumResults())
          throw py::index_error("Op result index out of range");
        return self->getResult(idx);
      });

  // Mirror the fork's builder.create_extern_call. Marshals libtriton-registered
  // pybind11 types and forwards to the builder; returns the OpState so
  // the Python frontend can read get_result(i) (0 results => void call, 1 =>
  // scalar return; >1 is rejected by the op's lowering).
  m.def("create_extern_call",
        [](TritonOpBuilder &self, const std::string &libName,
           const std::string &libPath, const std::string &symbol,
           const py::sequence &argSeq, std::vector<Type> &retTypes,
           bool isPure) -> OpState {
          std::vector<Value> argList = asValues(argSeq);
          return ::triton::dist::createExternCallOp(
              self, libName, libPath, symbol, argList, retTypes, isPure);
        });

  // SIMT execution region: a region-carrying op whose block args / results
  // thread the loop-carried (livein) values through the region so values
  // assigned inside it remain valid SSA outside. These cannot be OpInfo
  // callbacks (which only marshal Values); they return op/block handles the
  // Python frontend needs (mirrors the fork's
  // builder.create_simt_exec_region_op / get_simt_entry_block /
  // create_block_yield_op).
  m.def("create_simt_exec_region_op",
        [](TritonOpBuilder &self, const py::sequence &initArgSeq) -> OpState {
          std::vector<Value> initArgs = asValues(initArgSeq);
          return ::triton::dist::createSimtExecRegionOp(self, initArgs);
        });
  m.def(
      "get_simt_region_entry_block",
      [](OpState op) -> Block * {
        return ::triton::dist::getSimtRegionEntryBlock(op);
      },
      py::return_value_policy::reference);
  m.def("create_block_yield_op",
        [](TritonOpBuilder &self, const py::sequence &yieldSeq) -> OpState {
          std::vector<Value> yields = asValues(yieldSeq);
          return ::triton::dist::createBlockYieldOp(self, yields);
        });

  // --- Distributed / SIMT / shared-memory ops -------------------------------
  // Value-returning ops: the wrapper prepends the result slot, invokes the
  // builder, and returns the populated slot (see DIST_VALUE_OP above).
  DIST_VALUE_OP("distributed_wait", createDistributedWait);
  DIST_VALUE_OP("distributed_consume_token", createConsumeToken);
  DIST_VALUE_OP("distributed_get_rank", createGetRank);
  DIST_VALUE_OP("distributed_get_num_ranks", createGetNumRanks);
  DIST_VALUE_OP("distributed_symm_at", createSymmAt);
  DIST_VALUE_OP("distributed_extract", createExtract);
  DIST_VALUE_OP("distributed_insert", createInsert);
  DIST_VALUE_OP("distributed_load_shared", createLoadShared);
  DIST_VALUE_OP("distributed_memdesc_to_ptr", createMemDescToPtr);
  DIST_VALUE_OP("distributed_local_alloc", createLocalAlloc);
  DIST_VALUE_OP("distributed_memdesc_subview", createMemDescSubview);
  DIST_VALUE_OP("distributed_get_thread_id", createGetThreadId);
  DIST_VALUE_OP("distributed_get_block_size", createGetBlockSize);

  // Side-effect ops: no result is produced; the slot is still prepended so the
  // shared builder logic reads `operands[1..]` consistently.
  DIST_VOID_OP("distributed_notify", createNotify);
  DIST_VOID_OP("distributed_store_shared", createStoreShared);
  DIST_VOID_OP("distributed_local_dealloc", createLocalDealloc);
}
