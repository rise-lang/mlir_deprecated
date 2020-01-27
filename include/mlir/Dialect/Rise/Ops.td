//===- Ops.td - Loop operation definitions ---------------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines MLIR loop operations.
//
//===----------------------------------------------------------------------===//

#ifdef RISE_OPS
#else
#define RISE_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Rise_Dialect : Dialect {
let name = "rise";
let cppNamespace = "";
}

// Base class for Rise dialect ops.
class Rise_Op<string mnemonic, list<OpTrait> traits = []> :
Op<Rise_Dialect, mnemonic, traits> {
// For every rise op, there needs to be a:
//   * void print(OpAsmPrinter *p, ${C++ class of Op} op)
//   * LogicalResult verify(${C++ class of Op} op)
//   * ParseResult parse${C++ class of Op}(OpAsmParser *parser,
//                                         OperationState *result)
// functions.
//let printer = [{ return ::print(p, *this); }];
//let verifier = [{ return ::verify(*this); }];
//let parser = [{ return ::parse$cppClass(parser, result); }];
}


///Types
def NatType : Type<CPred<"$_self.isa<Nat>()">, "nat type">;

def IntType : Type<CPred<"$_self.isa<Int>()">, "int type">;
def FloatType : Type<CPred<"$_self.isa<Float>()">, "float type">;
def TupleType : Type<CPred<"$_self.isa<Tuple>()">, "tuple type">;
def ArrayType : Type<CPred<"$_self.isa<ArrayType>()">, "array type">;
def DataType : Type<Or<[IntType.predicate, FloatType.predicate, TupleType.predicate, ArrayType.predicate]>>;
def NatOrDataType : Type<Or<[NatType.predicate, DataType.predicate]>>;

def FunType : Type<CPred<"$_self.isa<FunType>()">, "fun type">;
def DataTypeWrapper : Type<CPred<"$_self.isa<DataTypeWrapper>()">, "data fun type">;
def RiseType : Type<Or<[FunType.predicate, DataTypeWrapper.predicate]>>;

def LambdaType : Type<CPred<"$_self.isa<LambdaType>()">, "lambda type">;
//def KindType : Type<CPred<"$_self.isa<Kind>()">, "kind type">;
def SubclassOfKindType : AnyTypeOf<[NatType, LambdaType], "subclass of kind type">;
//TODO: add other Types

///Attributes
def RiseTypeAttr : Attr<CPred<"$_self.isa<RiseTypeAttr>()">,
        "rise type attribute">{
    let storageType = [{ RiseTypeAttr }];
    let returnType = [{ mlir::Type }];
}
def DataTypeAttr : Attr<CPred<"$_self.isa<DataTypeAttr>()">,
        "datatype attribute">{
    let storageType = [{ DataTypeAttr }];
    let returnType = [{ mlir::Type }];
}
def NatAttr : Attr<CPred<"$_self.isa<NatAttr>()">,
        "nat attribute">{
    let storageType = [{ NatAttr }];
    let returnType = [{ int }];
}

def LiteralOp : Rise_Op<"literal", [NoSideEffect]> {
    let summary = "literal";
    let arguments = (ins AnyAttr:$literalValue);
    let results = (outs RiseType);
    let extraClassDeclaration = [{Attribute getValue() { return getAttr("value"); }
        /// Returns true if a constant operation can be built with the given value
        /// and result type.
        static bool isBuildableWith(Attribute value, mlir::Type type);
    }];
    let parser = [{ return parse$cppClass(parser, result); }];
}

def LambdaOp : Rise_Op<"lambda", [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "lambda";

    let regions = (region SizedRegion<1>:$region);
    let results = (outs FunType:$lambda_result);

    let parser = [{ return parse$cppClass(parser, result); }];
}

def ApplyOp : Rise_Op<"apply", [NoSideEffect]> {
    let summary = "apply";
    let arguments = (ins FunType:$fun, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$result);
    let parser = [{ return parse$cppClass(parser, result); }];
}

def MapOp : Rise_Op<"map", [NoSideEffect]> {
    let summary = "map";
    let arguments = (ins NatAttr:$n, DataTypeAttr:$s, DataTypeAttr:$t);
    let parser = [{ return parse$cppClass(parser, result); }];

    let results = (outs FunType:$mapFun);
}

def ReduceOp : Rise_Op<"reduce", [NoSideEffect]> {
    let summary = "reduce";
    let arguments = (ins NatAttr:$n, DataTypeAttr:$s, DataTypeAttr:$t);
    let parser = [{ return parse$cppClass(parser, result); }];

    let results = (outs FunType:$reduceFun);
}

def ZipOp : Rise_Op<"zip", [NoSideEffect]> {
    let summary = "rise zip operation";
    let description = [{
        }];
    let arguments = (ins NatAttr:$n, DataTypeAttr:$s, DataTypeAttr:$t);
    let parser = [{ return parse$cppClass(parser, result); }];
    let results = (outs FunType:$zipFun);
}

def TupleOp : Rise_Op<"tuple", [NoSideEffect]> {
    let summary = "rise tuple operation";
    let arguments = (ins DataTypeAttr:$s, DataTypeAttr:$t);
    let results = (outs FunType:$tupleFun);
    let parser = [{ return parse$cppClass(parser, result); }];
}

def FstOp: Rise_Op<"fst", [NoSideEffect]> {
    let summary = "rise fst operation";
    let arguments = (ins DataTypeAttr:$s, DataTypeAttr:$t);
    let results = (outs FunType:$fstFun);
    let parser = [{ return parse$cppClass(parser, result); }];
}

def SndOp: Rise_Op<"snd", [NoSideEffect]> {
    let summary = "rise snd operation";
    let arguments = (ins DataTypeAttr:$s, DataTypeAttr:$t);
    let results = (outs FunType:$sndFun);
    let parser = [{ return parse$cppClass(parser, result); }];
}

def ReturnOp : Rise_Op<"return", [Terminator, HasParent<"LambdaOp">]> {
    let summary = "rise terminator operation";
    let description = [{
        }];
    let arguments = (ins Variadic<AnyType>:$operands);
    let parser = [{ return parse$cppClass(parser, result); }];
    let builders = [OpBuilder<
        "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
        >];
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def AddOp : Rise_Op<"add", [NoSideEffect]> {
    let summary = "rise add operation";
    let arguments = (ins DataTypeAttr:$s);
    let results = (outs FunType:$addFun);
    let parser = [{ return parse$cppClass(parser, result); }];
}

def MultOp : Rise_Op<"mult", [NoSideEffect]> {
    let summary = "rise mult operation";
    let arguments = (ins DataTypeAttr:$s);
    let results = (outs FunType:$multFun);
    let parser = [{ return parse$cppClass(parser, result); }];
}

#endif // RISE_OPS
