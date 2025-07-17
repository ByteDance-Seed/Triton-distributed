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
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "DistributedDialect.td"
include "DistributedAttrDefs.td"

class TT_Distributed_Op<string mnemonic, list<Trait> traits = []> :
    Op<Distributed_Dialect, mnemonic, !listconcat(traits, [])> {
}


def TT_WaitOp : TT_Distributed_Op<"wait", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"result matches ptr type", "barrierPtr", "waitValue", "getPointeeType($_self)">,
]> {
  let summary = "Wait for a signal to be ready (becomes 1) and produce a token.";

  let description = [{
    The operator creates a wait node that represents a spin lock waiting
    for a list of signals pointed by the `barrierPtr` to be set ready (set to 1).

    Example:

    ```mlir
    token = distributed.wait() {barrierPtr, num_barriers, scope = 1 : i32, semantic = 0 : i32}
    ```
  }];
  let arguments = (
    ins
    TT_PtrLike:$barrierPtr,
    TT_IntLike:$numBarriers,
    TT_Int: $waitValue,
    DefaultValuedAttr<TT_MemSyncScopeAttr, "MemSyncScope::GPU">:$scope,
    DefaultValuedAttr<TT_MemSemanticAttr, "MemSemantic::ACQUIRE">:$semantic
  );

  let results = (outs TT_IntLike:$token);

  let assemblyFormat = [{
      $barrierPtr `,` $numBarriers `,` $waitValue `,`
      `scope` `=` $scope `semantic` `=` $semantic
      attr-dict `:` type($barrierPtr) `,` type($numBarriers) `,` type($waitValue) `->` type($token)
    }];
}

def TT_ConsumeTokenOp : TT_Distributed_Op<"consume_token", [
  Elementwise,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches input type", "input", "result", "$_self">,
]> {
  let summary = "Consumes a token and builds data dependency between wait ops and input";

  let description = [{
    The operator creates a consume node that represents a data dependency between a wait node
    that produces a token and another node that uses the input data (e.g., load).

    Example:

    ```mlir
    distributed.consume_token() {input, token}
    ```
  }];
  let arguments = (
    ins
    AnyTypeOf<[TT_Type, TT_TensorDescType]>:$input,
    TT_IntLike:$token
  );

  let results = (outs AnyTypeOf<[TT_Type, TT_TensorDescType]>:$result);

  let assemblyFormat = [{
      $input `,` $token `,`
      attr-dict `:` type($input) `,` type($token) `->` type($result)
    }];
}


// TODO(zhengxuegui.0): device mesh support
def TT_GetRankOp : TT_Distributed_Op<"get_rank", [Pure]> {
  let summary = "Rank of the calling PE";

  let arguments = (ins I32:$axis);

  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}


def TT_GetNumRanksOp : TT_Distributed_Op<"get_num_ranks", [Pure]> {
  let summary = "Get the number of PEs runing in the program";

  let arguments = (ins I32:$axis);

  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}


def TT_SymmAtOp : TT_Distributed_Op<"symm_at", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"remoteAddr matches symmAddr type", "symmAddr", "remoteAddr", "$_self">,
]> {
  let summary = "obtain the symmetric address on the specified Rank.";

  let arguments = (ins
    TT_Ptr:$symmAddr,
    I32:$rank
  );

  let results = (outs TT_Ptr:$remoteAddr);

  let assemblyFormat = "$symmAddr `,` $rank `,` attr-dict `:` qualified(type($symmAddr))";
}

def TT_NotifyOp : TT_Distributed_Op<"notify", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Update sigAddr with signalVal using operation sigOp on the specified rank";

  let arguments = (ins
    TT_Ptr:$sigAddr,
    I64:$signalVal,
    I32:$rank,
    TT_SignalOpAttr:$sigOp,
    TT_CommScopeAttr:$commScope
  );

}



def TT_ExternCallOp : TT_Distributed_Op<"extern_call", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  ConditionallySpeculatable,
]> {

    let description = [{
        call an external function $symbol implemented in $libpath/$libname with $args
        return $libpath/$libname:$symbol($args...)
    }];

    let arguments = (ins Variadic<TT_Type>:$srcs, StrAttr:$libname, StrAttr:$libpath, StrAttr:$symbol, BoolAttr:$pure);

    let results = (outs Variadic<TT_Type>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";

    let extraClassDeclaration = [{
      // Interface method for ConditionallySpeculatable.
      Speculation::Speculatability getSpeculatability();
    }];

}
#endif // DISTRIBUTED_OPS
