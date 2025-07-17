//
// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
#ifndef DISTRIBUTED_OPS
#define DISTRIBUTED_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike
include "SIMTDialect.td"
include "SIMTAttrDefs.td"

class TT_SIMT_Op<string mnemonic, list<Trait> traits = []> :
    Op<SIMT_Dialect, mnemonic, !listconcat(traits, [])> {
}

// SIMI Region related
def SIMTExecRegionOp : TT_SIMT_Op<"simt_exec_region", [
  RecursiveMemoryEffects,
]> {
  let summary = "perform the code in region with simt mode";
  let description = [{
    The `simt.simt_exec_region` op represents executing code in simt mode.
  }];

  let arguments = (ins Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$result);

  let regions = (region MinSizedRegion<1>:$defaultRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"ValueRange", "std::nullopt">:$initArgs)>
  ];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 0;
  let hasCanonicalizeMethod = 0;
}


def BlockYieldOp : TT_SIMT_Op<"block_yield", [
  Pure, Terminator, ReturnLike, HasParent<"SIMTExecRegionOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>
]> {
  let summary = "yield from the region of `simt.simt_exec_region`";
  let description = [{
    The `simt.block_yield` operation is the terminator for the region of
    a `simt.simt_exec_region` operation.

    Example:

    ```mlir
    simt.block_yield %a : tensor<128xbf16, #blocked>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "($values^)? attr-dict (`:` type($values)^)?";
  let hasVerifier = 1;
}


def LoadSharedOp : TT_SIMT_Op<"load_shared", [TypesMatchWith<"result type matches element type of 'src'",
                     "src", "result",
                     "::llvm::cast<mlir::triton::gpu::MemDescType>($_self).getElementType()">,
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load a value from shared memory into register";
  let description = [{
    Load a value from shared memory into register.
  }];

  let arguments = (ins TTG_MemDescType:$src, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$src `[` $indices `]` attr-dict `:` type($src)";

}

def StoreSharedOp : TT_SIMT_Op<"store_shared", [TypesMatchWith<"type of 'value' matches element type of 'dest'",
                     "dest", "value",
                     "::llvm::cast<mlir::triton::gpu::MemDescType>($_self).getElementType()">,
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store a value that in register to shared memory";
  let description = [{
    Store a value that in register to shared memory.
  }];

  let arguments = (ins AnyType:$value,
                  TTG_MemDescType:$dest,
                  Variadic<Index>:$indices);
  let assemblyFormat = [{
    $value `,` $dest `[` $indices `]` attr-dict `:` type($dest)
  }];
}
#endif // DISTRIBUTED_OPS
