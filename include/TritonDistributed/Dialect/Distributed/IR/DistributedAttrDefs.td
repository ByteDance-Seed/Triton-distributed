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
#ifndef DISTRIBUTED_ATTRDEFS
#define DISTRIBUTED_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "DistributedDialect.td"
include "mlir/IR/EnumAttr.td"

class Distributed_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<Distributed_Dialect, name, traits, baseCppClass> {
}


def TT_SignalOpAttr : I32EnumAttr<
    "SignalOp", "",
    [
        I32EnumAttrCase<"SET", 1, "set">,
        I32EnumAttrCase<"ADD", 2, "add">,
    ]> {
    let cppNamespace = "::mlir::triton::distributed";
}

def TT_CommScopeAttr : I32EnumAttr<
    "CommScope", "",
    [
      I32EnumAttrCase<"GPU", 1, "gpu">,
      I32EnumAttrCase<"INTRA_NODE", 2, "intra_node">,
      I32EnumAttrCase<"INTER_NODE", 3, "inter_node">,
    ]> {
    let cppNamespace = "::mlir::triton::distributed";
}
#endif // DISTRIBUTED_ATTRDEFS
