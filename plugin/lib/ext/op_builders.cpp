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

/// Distributed/SIMT op builders (create-op companion module).
///
/// Mirrors the fork's former `builder.create_*` pybind methods, re-expressed
/// against the upstream-style plugin ops. Enum attributes (MemSyncScope /
/// MemSemantic / SignalOp
/// / CommScope) are passed from Python as i32 constant Values, matching the
/// integer values in the dialect's TableGen enum definitions:
///   MemSyncScope : GPU=1, CTA=2, SYSTEM=3
///   MemSemantic  : RELAXED=1, ACQUIRE=2, RELEASE=3, ACQUIRE_RELEASE=4
///   SignalOp     : SET=1, ADD=2
///   CommScope    : GPU=1, INTRA_NODE=2, INTER_NODE=3
#include "op_builders.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"

#include <optional>

namespace triton::dist::ops {

namespace dist = mlir::triton::distributed;
namespace simt = mlir::triton::simt;
namespace ttg = mlir::triton::gpu;
using mlir::triton::MemSemantic;
using mlir::triton::MemSyncScope;

// Cast each index Value to `index` (the SIMT/tensor ops take
// `Variadic<Index>`).
static llvm::SmallVector<mlir::Value>
toIndexValues(TritonOpBuilder &self, llvm::ArrayRef<mlir::Value> vals) {
  llvm::SmallVector<mlir::Value> out;
  out.reserve(vals.size());
  auto idxTy = self.getBuilder().getIndexType();
  for (mlir::Value v : vals) {
    if (mlir::isa<mlir::IndexType>(v.getType()))
      out.push_back(v);
    else
      out.push_back(self.create<mlir::arith::IndexCastOp>(idxTy, v));
  }
  return out;
}

static std::optional<int64_t> extractConstantInt(mlir::Value v) {
  if (!v)
    return std::nullopt;
  if (auto c =
          mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(v.getDefiningOp()))
    return c.value();
  if (auto c =
          mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(v.getDefiningOp())) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

// operands: [result, barrierPtr, numBarriers, waitValue, scope(i32),
//            semantic(i32)]; the wait token is always i32 (matches the fork).
void createDistributedWait(TritonOpBuilder &self,
                           std::vector<Value> &operands) {
  if (operands.size() < 6)
    return;
  auto scope = extractConstantInt(operands[4]);
  auto semantic = extractConstantInt(operands[5]);
  if (!scope || !semantic)
    return;
  mlir::Type tokenType = self.getBuilder().getI32Type();
  // No predicate from the frontend (pred operand left empty).
  operands[0] = self.create<dist::WaitOp>(
      tokenType, operands[1], operands[2], operands[3],
      static_cast<MemSyncScope>(*scope), static_cast<MemSemantic>(*semantic),
      /*pred=*/mlir::Value());
}

// operands: [result, input, token]; result type == input type
void createConsumeToken(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 3)
    return;
  operands[0] = self.create<dist::ConsumeTokenOp>(operands[1], operands[2]);
}

// operands: [result, axis]
void createGetRank(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 2)
    return;
  operands[0] = self.create<dist::GetRankOp>(operands[1]);
}

// operands: [result, axis]
void createGetNumRanks(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 2)
    return;
  operands[0] = self.create<dist::GetNumRanksOp>(operands[1]);
}

// operands: [result, ptr, rank]; result type == ptr type
void createSymmAt(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 3)
    return;
  operands[0] = self.create<dist::SymmAtOp>(operands[1].getType(), operands[1],
                                            operands[2]);
}

// operands: [result(void), ptr, signal, rank, sigOp(i32), commScope(i32)]
void createNotify(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 6)
    return;
  auto sigOp = extractConstantInt(operands[4]);
  auto commScope = extractConstantInt(operands[5]);
  if (!sigOp || !commScope)
    return;
  self.create<dist::NotifyOp>(operands[1], operands[2], operands[3],
                              static_cast<dist::SignalOp>(*sigOp),
                              static_cast<dist::CommScope>(*commScope));
}

// operands: [result, input(tensor), idx...]; result = tensor element type
void createExtract(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 2)
    return;
  auto indices =
      toIndexValues(self, llvm::ArrayRef<Value>(operands).drop_front(2));
  operands[0] = self.create<mlir::tensor::ExtractOp>(operands[1], indices);
}

// operands: [result, scalar, dest(tensor), idx...]; result = dest type
void createInsert(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 3)
    return;
  auto indices =
      toIndexValues(self, llvm::ArrayRef<Value>(operands).drop_front(3));
  operands[0] =
      self.create<mlir::tensor::InsertOp>(operands[1], operands[2], indices);
}

// operands: [result, memdesc, idx...]; result = memdesc element type
void createLoadShared(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 2)
    return;
  auto memDescTy = mlir::cast<ttg::MemDescType>(operands[1].getType());
  auto indices =
      toIndexValues(self, llvm::ArrayRef<Value>(operands).drop_front(2));
  operands[0] = self.create<simt::LoadSharedOp>(memDescTy.getElementType(),
                                                operands[1], indices);
}

// operands: [result(void), value, memdesc, idx...]
void createStoreShared(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 3)
    return;
  auto indices =
      toIndexValues(self, llvm::ArrayRef<Value>(operands).drop_front(3));
  self.create<simt::StoreSharedOp>(operands[1], operands[2], indices);
}

// operands: [result, memdesc, addrSpace(i32), idx...]; result ptr elem type =
// memdesc element type, in the requested address space.
void createMemDescToPtr(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 3)
    return;
  auto addrSpace = extractConstantInt(operands[2]);
  if (!addrSpace)
    return;
  auto memDescTy = mlir::cast<ttg::MemDescType>(operands[1].getType());
  auto ptrTy = mlir::triton::PointerType::get(memDescTy.getElementType(),
                                              static_cast<int>(*addrSpace));
  auto indices =
      toIndexValues(self, llvm::ArrayRef<Value>(operands).drop_front(3));
  operands[0] = self.create<simt::MemDescToPtrOp>(ptrTy, operands[1], indices);
}

// operands: [result(void), memdesc]
void createLocalDealloc(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 2)
    return;
  self.create<ttg::LocalDeallocOp>(operands[1]);
}

// Build a swizzled shared-memory MemDescType (mirrors the fork's
// get_swizzled_shared_layout + get_shared_mem_desc_ty helpers).
static ttg::MemDescType makeSwizzledMemDescType(TritonOpBuilder &self,
                                                mlir::Type elemTy,
                                                llvm::ArrayRef<int64_t> shape,
                                                int vec, int perPhase,
                                                int maxPhase) {
  auto *ctx = self.getContext();
  unsigned rank = shape.size();
  // The shared encoding describes the *minor* (innermost) dims; leading dims
  // are treated as buffering (gluon's multibuffer convention). A rank-1 alloc
  // keeps a rank-1 encoding; rank>1 uses rank-1 so a single `memdesc_index`
  // (which requires an *identical* parent/child encoding) stays well-formed:
  // parent shape rank R with encoding rank R-1, child shape rank R-1 with the
  // same encoding -- both satisfy MemDescType's rank invariant.
  unsigned encRank = rank > 1 ? rank - 1 : 1;
  llvm::SmallVector<unsigned> order(encRank);
  for (unsigned i = 0; i < encRank; ++i)
    order[i] = encRank - 1 - i; // row-major: last dim fastest
  // Trivial single-CTA CGA layout (no block-level splitting): an empty set of
  // block bases over `encRank` standard out-dims. Mirrors gluon's
  // buildCgaLayoutAttr with empty cgaBases.
  auto kBlock = mlir::StringAttr::get(ctx, "block");
  mlir::triton::LinearLayout::BasesT bases;
  bases[kBlock] = {};
  auto outDims = mlir::triton::standardOutDimNames(ctx, encRank);
  mlir::triton::LinearLayout cgaLL(std::move(bases), outDims);
  auto cgaLayout = ttg::CGAEncodingAttr::get(ctx, std::move(cgaLL));
  auto enc = ttg::SwizzledSharedEncodingAttr::get(ctx, vec, perPhase, maxPhase,
                                                  order, cgaLayout);
  return ttg::MemDescType::get(shape, elemTy, enc,
                               ttg::SharedMemorySpaceAttr::get(ctx),
                               /*mutableMemory=*/true,
                               /*allocShape=*/shape);
}

// operands: [result, elemTypeCarrier, vec, perPhase, maxPhase, shape...]
// (alloc_shape == shape, matching allocate_smem). The element type is carried
// by the type of a (dead) scalar value of that element type.
void createLocalAlloc(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 6)
    return;
  auto vec = extractConstantInt(operands[2]);
  auto perPhase = extractConstantInt(operands[3]);
  auto maxPhase = extractConstantInt(operands[4]);
  if (!vec || !perPhase || !maxPhase)
    return;
  mlir::Type elemTy = operands[1].getType();
  llvm::SmallVector<int64_t> shape;
  for (size_t i = 5; i < operands.size(); ++i) {
    auto d = extractConstantInt(operands[i]);
    if (!d)
      return;
    shape.push_back(*d);
  }
  auto memDescTy =
      makeSwizzledMemDescType(self, elemTy, shape, *vec, *perPhase, *maxPhase);
  operands[0] = self.create<ttg::LocalAllocOp>(memDescTy);
}

// operands: [result, parentMemdesc, index]; indexes the leading dim, dropping
// it (mirrors smem_index: sub-shape = parentShape[1:], swizzle 1/1/1). On the
// pinned Triton 3.7.1 this is ttg.memdesc_index (was memdesc_subview in the
// fork).
void createMemDescSubview(TritonOpBuilder &self, std::vector<Value> &operands) {
  if (operands.size() < 3)
    return;
  auto parentTy = mlir::cast<ttg::MemDescType>(operands[1].getType());
  auto parentShape = parentTy.getShape();
  llvm::SmallVector<int64_t> subShape(parentShape.begin() + 1,
                                      parentShape.end());
  // memdesc_index keeps the parent's shared encoding (it is shape-independent);
  // only the logical shape (and alloc shape) drop the leading dim. Mirrors
  // gluon's GluonSemantic.memdesc_index.
  auto subTy = ttg::MemDescType::get(
      subShape, parentTy.getElementType(), parentTy.getEncoding(),
      parentTy.getMemorySpace(), parentTy.getMutableMemory(),
      /*allocShape=*/subShape);
  operands[0] =
      self.create<ttg::MemDescIndexOp>(subTy, operands[1], operands[2]);
}

// --- SIMT thread/block intrinsics ------------------------------------------
//
// The `simt.simt_exec_region` op itself (which carries a region and threads
// loop-carried values through block args/results) is built via
// createSimtExecRegionOp/createBlockYieldOp (simt_region_builder.cpp), since it
// must hand op/block handles back to the frontend. The thread-id / block-size
// intrinsics below are ordinary value-producing ops.

namespace gpu_dialect = ::mlir::gpu;

// Per-thread linear index within the block: gpu.thread_id x, cast to i32.
// gpu.* stays legal through the Distributed->LLVM pass and is lowered by the
// NVIDIA pipeline's GPU->NVVM step (same path the fork relied on).
void createGetThreadId(TritonOpBuilder &self, std::vector<Value> &operands) {
  mlir::Value tid =
      self.create<gpu_dialect::ThreadIdOp>(gpu_dialect::Dimension::x);
  operands[0] = self.create<mlir::arith::IndexCastOp>(
      self.getBuilder().getIntegerType(32), tid);
}

// Threads per block along x: gpu.block_dim x, cast to i32.
void createGetBlockSize(TritonOpBuilder &self, std::vector<Value> &operands) {
  mlir::Value bs =
      self.create<gpu_dialect::BlockDimOp>(gpu_dialect::Dimension::x);
  operands[0] = self.create<mlir::arith::IndexCastOp>(
      self.getBuilder().getIntegerType(32), bs);
}

} // namespace triton::dist::ops
