//
// Created by martin on 2019-09-23.
//
#include "mlir/Dialect/Rise/Ops.h"

#include "mlir/Dialect/Rise/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////
using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;


namespace mlir {
namespace rise {

//===----------------------------------------------------------------------===//
// LambdaOp
//===----------------------------------------------------------------------===//
ParseResult parseLambdaOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

//    OpAsmParser::OperandType lambdaInputVariable;
    SmallVector<OpAsmParser::OperandType, 10> inputs;
    SmallVector<Type, 10> inputTypes = SmallVector<Type, 10>();
    FunType funType;
//    RiseType lambdaInputType;
//    RiseType lambdaOutputType;

    // Parse the lambdaInput variable
//    if (parser.parseRegionArgument(lambdaInputVariable))
//        return failure();
    if (parser.parseRegionArgumentList(inputs, OpAsmParser::Delimiter::Paren))
        failure();

    //parse LambdaInputType
    if (parser.parseColon() || parser.parseType(funType))
        return failure();

//    std::cout << "yo: " << RiseDialect::stringForType(funType);
    ///outsource to its own method
    inputTypes.push_back(funType.getInput());

    //handle multiple inputs:
    for (int i = 1; i < inputs.size(); i++) {
        if (funType.getOutput().isa<FunType>()) {
            funType = funType.getOutput().dyn_cast<FunType>();
            inputTypes.push_back(funType.getInput());
        } else {
            parser.emitError(parser.getCurrentLocation()) << ": number of arguments: " << std::to_string(i) << " is too high for specified funType";
            return failure();
        }
    }

    //parse LambdaOutputType
//    if (parser.parseArrow() || parser.parseType(lambdaOutputType))
//        return failure();

    // Parse the body region.
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, inputs, inputTypes))
        return failure();
    LambdaOp::ensureTerminator(*body, builder, result.location);

    // Parse the optional attribute list.
//    if (parser.parseOptionalAttributeDict(result.attributes))
//        return failure();

//    RiseType funInput;
//    RiseType funOutput;

    //This seems not like the right way to do this.
//    switch (lambdaInputType.getKind()) {
//        default: {
//            funInput = lambdaInputType;
//            break;
//        }
//        case RiseTypeKind::RISE_INT: {
//            funInput = DataTypeWrapper::get(builder.getContext(), lambdaInputType.dyn_cast<Int>());
//            break;
//        }
//        case RiseTypeKind::RISE_FLOAT: {
//            funInput = DataTypeWrapper::get(builder.getContext(), lambdaInputType.dyn_cast<Float>());
//            break;
//        }
//    }
//    switch (lambdaOutputType.getKind()) {
//        default: {
//            funOutput = lambdaOutputType;
//            break;
//        }
//        case RiseTypeKind::RISE_INT: {
//            funOutput = DataTypeWrapper::get(builder.getContext(), lambdaOutputType.dyn_cast<Int>());
//            break;
//        }
//        case RiseTypeKind::RISE_FLOAT: {
//            funOutput = DataTypeWrapper::get(builder.getContext(), lambdaOutputType.dyn_cast<Float>());
//            break;
//        }
//    }

//    FunType type = FunType::get(builder.getContext(), funInput, funOutput);
    result.addTypes(funType);
    return success();
}

//LogicalResult LambdaOp::verify() {
//        unsigned index = 0; (void)index;
//        for (Value *v : getODSResults(0)) {
//            (void)v;
//            if (!((v.getType().isa<FunType>()))) {
//                return emitOpError("result #") << index << " must be funny type, but got " << v.getType();
//            }
//            ++index;
//        }
//    if (this->getOperation()->getNumRegions() != 1) {
//        return emitOpError("has incorrect number of regions: expected 1 but found ") << this->getOperation()->getNumRegions();
//    }
//    if (!((this->getOperation()->getRegion(0).getBlocks().size() == 1))) {
//        return emitOpError("region #0 ('region') failed to verify constraint: region with 1 blocks");
//    }
//    return mlir::success();
//}


//ArrayType parseArrayLiteral()

//===----------------------------------------------------------------------===//
// ParseLiteralOp
//===----------------------------------------------------------------------===//
ParseResult parseLiteralOp(OpAsmParser &parser, OperationState &result) {
    ///Format:
    ///         rise.literal #rise.int<42>
    ///         rise.literal #rise.array<2, rise.int, [1,2]>
    ///         rise.literal #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>

    auto &builder = parser.getBuilder();


//    if (parser.parseType(literalType))
//        failure();

//    if (parser.parseLParen())
//        failure();

    LiteralAttr attr;
    if (parser.parseAttribute(attr, "literalValue",result.attributes))
        return failure();


//    switch (attr.getType().getKind()) {
//        default: {
//            return failure();
//        }
//        //If subclass of DataType we have to wrap
//        case RiseTypeKind::RISE_INT: {
//            literalType = DataTypeWrapper::get(builder.getContext(), attr.getType().dyn_cast<Int>());
//            break;
//        }
//        case RiseTypeKind::RISE_FLOAT: {
//            literalType = DataTypeWrapper::get(builder.getContext(), attr.getType().dyn_cast<Float>());
//            break;
//        }
//        case RiseTypeKind::RISE_ARRAY: {
//            literalType = DataTypeWrapper::get(builder.getContext(), attr.getType().dyn_cast<ArrayType>());
//            break;
//        }
//        case RiseTypeKind::RISE_TUPLE: {
//            literalType = DataTypeWrapper::get(builder.getContext(), attr.getType().dyn_cast<Tuple>());
//            break;
//        }
//        //If subclass of RiseType we can use it directly
//        case RiseTypeKind::RISE_FUNTYPE: {
//            literalType = attr.getType().dyn_cast<FunType>();
//        }
//        case RiseTypeKind::RISE
//    }


    result.addTypes(DataTypeWrapper::get(builder.getContext(), attr.getType()));

    return success();


    if (succeeded(parser.parseOptionalLSquare())) {
        //Houston, we have an array

        ///These should be parsable
        ///%0 = rise.array [1,5,6,1,0]
        ///%0 = rise.array [[1,2],[2,3]]

//        if (parser.parseComm)
//        while (succeeded(parser.parseOptionalComma())) {
//
//        }

    } else {
        //    OpAsmParser::OperandType value;
////    Attribute value;
        mlir::Type literalType;
//
//
//    if (parser.parseOperand(value))
//        return failure();
        Attribute attr;
        if (parser.parseAttribute(attr, IntegerType::get(16, parser.getBuilder().getContext()), "literalValue",result.attributes))
            return failure();


        if (parser.parseColonType(literalType))
            return failure();


//    if(parser.parseAttribute(value, "literalValue", result.attributes))
//        return failure();

//    auto attr = builder.getTypeAttr(literalType);
//    result.addAttribute("literalValue", attr);
//    if (parser.parse)
//    if(parser.parseColon())
//        return failure();

//    result.addAttribute("literalValue", value);
        result.addTypes(literalType);

    }


    return success();
}


//===----------------------------------------------------------------------===//
// ParseArrayOp
//===----------------------------------------------------------------------===//
///// to be replaced by literalop
//ParseResult parseArrayOp(OpAsmParser &parser, OperationState &result) {
//    auto &builder = parser.getBuilder();
//
////    IntegerAttr sizeAttr;
////    RiseTypeAttr elementTypeAttr;
////
////    if (parser.parseAttribute(sizeAttr,
////            IntegerType::get(16, parser.getBuilder().getContext()),
////            "size",result.attributes))
////        return failure();
////
//////    if (parser.parseAttribute(elementTypeAttr, Type(),
//////            "elementType", result.attributes))
//////        return failure();
////    ///TODO: this should really be done using parseAttribute, but for some reason the method parseAttribute of this dialect is not called
////    mlir::Type type;
////    if (parser.parseType(type))
////        failure();
////
////    elementTypeAttr = RiseTypeAttr::get(type);
////    result.addAttribute("elementType", elementTypeAttr);
////
////    //TODO change hardcode of Nat
////    //somewhere the nat elementType gets changed to a float
////    auto arrayType = ArrayType::get(parser.getBuilder().getContext(),
////            sizeAttr.getInt(), type);
////
////    result.addTypes(arrayType);
//
//    return success();
//}


//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//
ParseResult parseApplyOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    OpAsmParser::OperandType funOperand;
    FunType funType;

    SmallVector<OpAsmParser::OperandType, 10> inputs;
    SmallVector<Type, 10> inputTypes = SmallVector<Type,10>();

    OpAsmParser::OperandType argumentOperand;
    Type argumentType;
    Type outputType;

    result.setOperandListToResizable();

    //now looks %result = rise.apply %id : !rise.fun<!rise.data<int> -> !rise.data<int>>, %42
    //parse function
    if (parser.parseOperand(funOperand))
        return failure();
    if (parser.parseColonType(funType))
        return failure();

    ///resolve operand adds it to the operands of this operation. I have not found another way to add it, yet
    ///addOperands expects a Value, which has to contain the Type of the Operand already, which I don't know here
    if (parser.resolveOperand(funOperand, funType, result.operands))
        failure();
    //    Value()
    //    result.addOperands(funOperand);

    //parse inputs
    if (parser.parseTrailingOperandList(inputs))
        failure();

    //one input is always needed
    inputTypes.push_back(funType.getInput());

    //handle multiple inputs:
    for (int i = 1; i < inputs.size(); i++) {
        if (funType.getOutput().isa<FunType>()) {
            funType = funType.getOutput().dyn_cast<FunType>();
            inputTypes.push_back(funType.getInput());
        } else {
            parser.emitError(parser.getCurrentLocation()) << "expected a maximum " << std::to_string(i) << " inputs for this function.";
            return failure();
        }
    }

    if (parser.resolveOperands(inputs, inputTypes, parser.getCurrentLocation(), result.operands))
        failure();

//    if (parser.resolveOperand(inputs.front(), argumentType, result.operands))
//        failure();


    result.addTypes(funType.getOutput());
    return success();
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    OpAsmParser::OperandType value;
    Type type;
    result.setOperandListToResizable();

//    SmallVector<OpAsmParser::OperandType, 1> values;
//    if (parser.parseOperandList(values, 0))
//        return failure();
//
//    if (parser.resolveOperands(values, {Nat::get(parser.getBuilder().getContext())}, result.operands))
//        return failure();


    if (parser.parseOperand(value))
        failure();
    if (parser.parseColonType(type))
        failure();
    if (parser.resolveOperand(value, type, result.operands))
        failure();

    return success();
}

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//
/// (n : nat) → (s t : data) → (exp[s] → exp[t ]) → exp[n.s] → exp[n.t ]
ParseResult parseMapOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    NatAttr n;
    DataTypeAttr s, t;
    result.setOperandListToResizable();

    if (parser.parseAttribute(n, "n", result.attributes))
        failure();

    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    if (parser.parseAttribute(t, "t", result.attributes))
        failure();

    result.addTypes(FunType::get(builder.getContext(),
                                 FunType::get(builder.getContext(),
                                         DataTypeWrapper::get(builder.getContext(), s.getValue()),
                                         DataTypeWrapper::get(builder.getContext(), t.getValue())),
                                 FunType::get(builder.getContext(),
                                              DataTypeWrapper::get(builder.getContext(), ArrayType::get(builder.getContext(), n.getValue(), s.getValue())),
                                              DataTypeWrapper::get(builder.getContext(), ArrayType::get(builder.getContext(), n.getValue(), t.getValue())))));

    return success();
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//
ParseResult parseReduceOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    NatAttr n;
    DataTypeAttr s, t;
    result.setOperandListToResizable();

    //n - number of elements in Array
    if (parser.parseAttribute(n, "n", result.attributes))
        failure();

    //elementType of Array
    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    //resulttype and initializer type
    if (parser.parseAttribute(t, "t", result.attributes))
        failure();
    /// (n : nat) → (s t : data) → (exp[s] → exp[t ] → exp[t ]) → exp[t ] → exp[n.s] → exp[t ]

    result.addTypes(FunType::get(builder.getContext(),
                                 FunType::get(builder.getContext(),
                                              DataTypeWrapper::get(builder.getContext(), s.getValue()),
                                              FunType::get(builder.getContext(),
                                                           DataTypeWrapper::get(builder.getContext(), t.getValue()),
                                                           DataTypeWrapper::get(builder.getContext(), t.getValue()))),
                                 FunType::get(builder.getContext(),
                                              DataTypeWrapper::get(builder.getContext(), t.getValue()),
                                              FunType::get(builder.getContext(),
                                                      DataTypeWrapper::get(builder.getContext(), ArrayType::get(builder.getContext(), n.getValue(), s.getValue())),
                                                      DataTypeWrapper::get(builder.getContext(), t.getValue())))));

    return success();
}

//===----------------------------------------------------------------------===//
// Tuples
//===----------------------------------------------------------------------===//
ParseResult parseTupleOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    DataTypeAttr s, t;
    result.setOperandListToResizable();

    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    if (parser.parseAttribute(t, "t", result.attributes))
        failure();

    result.addTypes(FunType::get(builder.getContext(),
            DataTypeWrapper::get(builder.getContext(), s.getValue()),
            FunType::get(builder.getContext(),
                    DataTypeWrapper::get(builder.getContext(), t.getValue()),
                    DataTypeWrapper::get(builder.getContext(), Tuple::get(builder.getContext(), s.getValue(), t.getValue())))));

    return success();
}

ParseResult parseFstOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    DataTypeAttr s, t;
    result.setOperandListToResizable();

    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    if (parser.parseAttribute(t, "t", result.attributes))
        failure();

    result.addTypes(FunType::get(builder.getContext(),
                                 DataTypeWrapper::get(builder.getContext(), Tuple::get(builder.getContext(), s.getValue(), t.getValue())),
                                 DataTypeWrapper::get(builder.getContext(), s.getValue())));

    return success();
}

ParseResult parseSndOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    DataTypeAttr s, t;
    result.setOperandListToResizable();

    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    if (parser.parseAttribute(t, "t", result.attributes))
        failure();

    result.addTypes(FunType::get(builder.getContext(),
                                 DataTypeWrapper::get(builder.getContext(),
                                         Tuple::get(builder.getContext(), s.getValue(), t.getValue())),
                                 DataTypeWrapper::get(builder.getContext(), t.getValue())));

    return success();
}

///zip: (n : nat) → (s t : data) → exp[n.s] → exp[n.t ] → exp[n.(s × t )]

ParseResult parseZipOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    NatAttr n;
    DataTypeAttr s, t;

    if (parser.parseAttribute(n, "n",result.attributes))
        return failure();

    if (parser.parseAttribute(s, "s",result.attributes))
        return failure();

    if (parser.parseAttribute(t, "t",result.attributes))
        return failure();

    result.addTypes(FunType::get(builder.getContext(),
            DataTypeWrapper::get(builder.getContext(), ArrayType::get(builder.getContext(), n.getValue(), s.getValue())),
                    FunType::get(builder.getContext(),
                            DataTypeWrapper::get(builder.getContext(), ArrayType::get(builder.getContext(), n.getValue(), t.getValue())),
                            DataTypeWrapper::get(builder.getContext(), ArrayType::get(builder.getContext(), n.getValue(), Tuple::get(builder.getContext(), s.getValue(), t.getValue()))))));

    return success();

}

//===----------------------------------------------------------------------===//
// Arithmetics
//===----------------------------------------------------------------------===//


ParseResult parseAddOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    DataTypeAttr s;

    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    result.addTypes(FunType::get(builder.getContext(),
                                 DataTypeWrapper::get(builder.getContext(), s.getValue()),
                                 FunType::get(builder.getContext(),
                                         DataTypeWrapper::get(builder.getContext(), s.getValue()),
                                         DataTypeWrapper::get(builder.getContext(), s.getValue()))));

    return success();
}

ParseResult parseMultOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    DataTypeAttr s;

    if (parser.parseAttribute(s, "s", result.attributes))
        failure();

    result.addTypes(FunType::get(builder.getContext(),
                                 DataTypeWrapper::get(builder.getContext(), s.getValue()),
                                 FunType::get(builder.getContext(),
                                              DataTypeWrapper::get(builder.getContext(), s.getValue()),
                                              DataTypeWrapper::get(builder.getContext(), s.getValue()))));

    return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Rise/Ops.cpp.inc"
} //end namespace rise
} //end namespace mlir


