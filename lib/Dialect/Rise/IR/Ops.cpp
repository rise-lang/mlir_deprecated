//
// Created by martin on 2019-09-23.
//
#include "mlir/Dialect/Rise/Ops.h"

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

    OpAsmParser::OperandType lambdaInputVariable;
    RiseType lambdaInputType;
    RiseType lambdaOutputType;
    // Parse the lambdaInput variable
    if (parser.parseRegionArgument(lambdaInputVariable))
        return failure();

    //parse LambdaInputType
    if (parser.parseColon() || parser.parseType(lambdaInputType))
        return failure();

    //parse LambdaOutputType
    if (parser.parseArrow() || parser.parseType(lambdaOutputType))
        return failure();

    // Parse the body region.
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, lambdaInputVariable, lambdaInputType))
        return failure();
    LambdaOp::ensureTerminator(*body, builder, result.location);

    // Parse the optional attribute list.
    if (parser.parseOptionalAttributeDict(result.attributes))
        return failure();
    std::cout << "yo \n";

    RiseType funInput;
    RiseType funOutput;

    //This seems not like the right way to do this.
    switch (lambdaInputType.getKind()) {
        default: {
            funInput = lambdaInputType;
            break;
        }
        case RiseTypeKind::RISE_INT: {
            funInput = DataTypeWrapper::get(builder.getContext(), lambdaInputType.dyn_cast<Int>());
            break;
        }
        case RiseTypeKind::RISE_FLOAT: {
            funInput = DataTypeWrapper::get(builder.getContext(), lambdaInputType.dyn_cast<Float>());
            break;
        }
    }
    switch (lambdaOutputType.getKind()) {
        default: {
            funOutput = lambdaOutputType;
            break;
        }
        case RiseTypeKind::RISE_INT: {
            funOutput = DataTypeWrapper::get(builder.getContext(), lambdaOutputType.dyn_cast<Int>());
            break;
        }
        case RiseTypeKind::RISE_FLOAT: {
            funOutput = DataTypeWrapper::get(builder.getContext(), lambdaOutputType.dyn_cast<Float>());
            break;
        }
    }

    FunType type = FunType::get(builder.getContext(), funInput, funOutput);
    result.addTypes(type);
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

    RiseTypeAttr attr;
    if (parser.parseAttribute(attr, "literalValue",result.attributes))
        return failure();

    result.addTypes(attr.getType());

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
    OpAsmParser::OperandType argumentOperand;
    Type argumentType;
    Type outputType;

    //now looks %result = rise.apply %id : !rise.fun<!rise.wrapped<int>, !rise.wrapped<int>>, %42
    //parse LambdaInputType
    if (parser.parseOperand(funOperand))
        return failure();
    if (parser.parseColonType(funType))
        return failure();

//    FunType funInputType = FunType::get(builder.getContext(), DataTypeWrapper::get(builder.getContext(), Int::get(builder.getContext())), DataTypeWrapper::get(builder.getContext(), Int::get(builder.getContext())));  //funInputOperand.dyn_cast<FunType>();
    if (parser.resolveOperand(funOperand, funType, result.operands))
        failure();

    //parse LambdaOutputType
    if (parser.parseComma() || parser.parseOperand(argumentOperand))
        return failure();
    argumentType = funType.getInput();

    //unpacking, if result is in a wrapper. This way the 2nd operand and the result of apply is never in a wrapper
    //TODO: should we really unpack here?
    if (argumentType.isa<DataTypeWrapper>()) {
        argumentType = argumentType.dyn_cast<DataTypeWrapper>().getData();
    }
    if (parser.resolveOperand(argumentOperand, argumentType, result.operands))
        failure();

    outputType = funType.getOutput();
    if (outputType.isa<DataTypeWrapper>()) {
        outputType = outputType.dyn_cast<DataTypeWrapper>().getData();
    }

    result.setOperandListToResizable();
    result.addTypes(outputType);
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

static void print(OpAsmPrinter *p, LiteralOp &op) {
    *p << "literal ";
    p->printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

    if (op.getAttrs().size() > 1)
        *p << ' ';
    p->printAttribute(op.getValue());

    // If the value is a symbol reference, print a trailing type.
    if (op.getValue().isa<SymbolRefAttr>())
        *p << " : " << op.getType();
}

//===----------------------------------------------------------------------===//
// Tuples
//===----------------------------------------------------------------------===//
ParseResult parseTupleOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    OpAsmParser::OperandType left;
    OpAsmParser::OperandType right;
    DataType leftType;
    DataType rightType;

    result.setOperandListToResizable();


    if (parser.parseOperand(left) || parser.parseColon() || parser.parseType(leftType))
        failure();
    if (parser.parseComma())
        failure();
    if (parser.parseOperand(right) || parser.parseColon() || parser.parseType(rightType))
        failure();

    if (parser.resolveOperand(left, leftType, result.operands))
        failure();
    if (parser.resolveOperand(right, rightType, result.operands))
        failure();
    result.addTypes(Tuple::get(builder.getContext(), leftType, rightType));

    return success();
}

ParseResult parseZipOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    OpAsmParser::OperandType left;
    OpAsmParser::OperandType right;
    ArrayType leftType;
    ArrayType rightType;

    result.setOperandListToResizable();


    if (parser.parseOperand(left) || parser.parseColonType(leftType))
        failure();
    if (parser.resolveOperand(left, leftType, result.operands))
        failure();

    if (parser.parseComma())
        failure();

    if (parser.parseOperand(right) || parser.parseColonType(rightType))
        failure();
    if (parser.resolveOperand(right, rightType, result.operands))
        failure();

    if (rightType != leftType)
        std::cout << "Arrays are not compatible";

    ArrayType resultType = ArrayType::get(builder.getContext(), rightType.getSize(),
            Tuple::get(builder.getContext(), leftType.getElementType(), rightType.getElementType()));
    result.addTypes(resultType);

    return success();
}

//===----------------------------------------------------------------------===//
// Arithmetics
//===----------------------------------------------------------------------===//

ParseResult parseAddIntOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    OpAsmParser::OperandType summand0;
    OpAsmParser::OperandType summand1;
    result.setOperandListToResizable();


    if (parser.parseOperand(summand0) || parser.parseComma() || parser.parseOperand(summand1))
        failure();
    if (parser.resolveOperand(summand0, Int::get(builder.getContext()), result.operands))
        failure();
    if (parser.resolveOperand(summand1, Int::get(builder.getContext()), result.operands))
        failure();

    result.addTypes(Int::get(builder.getContext()));

    return success();
}

ParseResult parseAddFloatOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();

    OpAsmParser::OperandType summand0;
    OpAsmParser::OperandType summand1;
    result.setOperandListToResizable();


    if (parser.parseOperand(summand0) || parser.parseComma() || parser.parseOperand(summand1))
        failure();
    if (parser.resolveOperand(summand0, Float::get(builder.getContext()), result.operands))
        failure();
    if (parser.resolveOperand(summand1, Float::get(builder.getContext()), result.operands))
        failure();

    result.addTypes(Float::get(builder.getContext()));

    return success();
}

//    static void print(OpAsmPrinter *p, ApplyOp op) {
//        *p << "call " << op.getAttr("callee") << '(';
//        p.printOperands(op.getOperands());
//        *p << ')';
////        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
////        *p << " : ";
////        p.printType(op.getCalleeType());
//    }

/// Helper to verify that the result of an operation is a Rise array type.
//template<typename T>
//static mlir::LogicalResult verifyRiseReturnArray(T *op) {
//    if (!op.getResult().getType().template isa<RiseArrayType>()) {
//        std::string msg;
//        raw_string_ostream os(msg);
//        os << "expects a Rise Array for its argument, got "
//           << op->getResult()->getType();
//        return op->emitOpError(os.str());
//    }
//    return mlir::success();
//}

///// Helper to verify that the two operands of a binary operation are Rise
///// arrays..
//template<typename T>
//static mlir::LogicalResult verifyRiseBinOperands(T *op) {
//    if (!op->getOperand(0)->getType().template isa<RiseArrayType>()) {
//        std::string msg;
//        raw_string_ostream os(msg);
//        os << "expects a Rise Array for its LHS, got "
//           << op->getOperand(0)->getType();
//        return op->emitOpError(os.str());
//    }
//    if (!op->getOperand(1)->getType().template isa<RiseArrayType>()) {
//        std::string msg;
//        raw_string_ostream os(msg);
//        os << "expects a Rise Array for its LHS, got "
//           << op->getOperand(0)->getType();
//        return op->emitOpError(os.str());
//    }
//    return mlir::success();
//}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
//void ConstantOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                       ArrayRef<int64_t> shape, mlir::DenseElementsAttr value) {
//    state->types.push_back(RiseArrayType::get(builder->getContext(), shape));
//    auto dataAttribute = builder->getNamedAttr("value", value);
//    state->attributes.push_back(dataAttribute);
//}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
//void ConstantOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                       mlir::FloatAttr value) {
//    // Broadcast and forward to the other build factory
//    mlir::Type elementType = mlir::FloatType::getF64(builder->getContext());
//    auto dataType = builder->getTensorType({1}, elementType);
//    auto dataAttribute = builder->getDenseElementsAttr(dataType, {value})
//            .cast<mlir::DenseElementsAttr>();
//
//    ConstantOp::build(builder, state, {1}, dataAttribute);
//}

/// Verifier for constant operation.
//mlir::LogicalResult ConstantOp::verify() {
//    // Ensure that the return type is a Rise array
//    if (failed(verifyRiseReturnArray(this)))
//        return mlir::failure();
//
//    // We expect the constant itself to be stored as an attribute.
//    auto dataAttr = getAttr("value").dyn_cast<mlir::DenseElementsAttr>();
//    if (!dataAttr) {
//        return emitOpError(
//                "missing valid `value` DenseElementsAttribute on rise.constant()");
//    }
//    auto attrType = dataAttr.getType().dyn_cast<mlir::TensorType>();
//    if (!attrType) {
//        return emitOpError(
//                "missing valid `value` DenseElementsAttribute on rise.constant()");
//    }
//
//    // If the return type of the constant is not a generic array, the shape must
//    // match the shape of the attribute holding the data.
//    auto resultType = getResult()->getType().cast<RiseArrayType>();
//    if (!resultType.isGeneric()) {
//        if (attrType.getRank() != resultType.getRank()) {
//            return emitOpError("The rank of the rise.constant return type must match "
//                               "the one of the attached value attribute: " +
//                               Twine(attrType.getRank()) +
//                               " != " + Twine(resultType.getRank()));
//        }
//        for (int dim = 0; dim < attrType.getRank(); ++dim) {
//            if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
//                std::string msg;
//                raw_string_ostream os(msg);
//                return emitOpError(
//                        "Shape mismatch between rise.constant return type and its "
//                        "attribute at dimension " +
//                        Twine(dim) + ": " + Twine(attrType.getShape()[dim]) +
//                        " != " + Twine(resultType.getShape()[dim]));
//            }
//        }
//    }
//    return mlir::success();
//}

//void ApplyOp::build(mlir::Builder *builder, mlir::OperationState *state, llvm::StringRef callee,
//                    llvm::ArrayRef<mlir::Value *> arguments) {
//
//
//}
//
//void GenericCallOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                          StringRef callee, ArrayRef<mlir::Value *> arguments) {
//    // Generic call always returns a generic RiseArray initially
//    state->types.push_back(RiseArrayType::get(builder->getContext()));
//    state->operands.assign(arguments.begin(), arguments.end());
//    auto calleeAttr = builder->getStringAttr(callee);
//    state->attributes.push_back(builder->getNamedAttr("callee", calleeAttr));
//}
//
//mlir::LogicalResult GenericCallOp::verify() {
//    // Verify that every operand is a Rise Array
//    for (int opId = 0, num = getNumOperands(); opId < num; ++opId) {
//        if (!getOperand(opId)->getType().template isa<RiseArrayType>()) {
//            std::string msg;
//            raw_string_ostream os(msg);
//            os << "expects a Rise Array for its " << opId << " operand, got "
//               << getOperand(opId)->getType();
//            return emitOpError(os.str());
//        }
//    }
//    return mlir::success();
//}

/// Return the name of the callee.
//StringRef GenericCallOp::getCalleeName() {
//    return getAttr("callee").cast<mlir::StringAttr>().getValue();
//}
//
//template<typename T>
//static mlir::LogicalResult verifyRiseSingleOperand(T *op) {
//    if (!op->getOperand()->getType().template isa<RiseArrayType>()) {
//        std::string msg;
//        raw_string_ostream os(msg);
//        os << "expects a Rise Array for its argument, got "
//           << op->getOperand()->getType();
//        return op->emitOpError(os.str());
//    }
//    return mlir::success();
//}

//void ReturnOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                     mlir::Value *value) {
//    // Return does not return any value and has an optional single argument
//    if (value)
//        state->operands.push_back(value);
//}
//
//mlir::LogicalResult ReturnOp::verify() {
//    if (getNumOperands() > 1) {
//        std::string msg;
//        raw_string_ostream os(msg);
//        os << "expects zero or one operand, got " << getNumOperands();
//        return emitOpError(os.str());
//    }
//    if (hasOperand() && failed(verifyRiseSingleOperand(this)))
//        return mlir::failure();
//    return mlir::success();
//}

//void PrintOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                    mlir::Value *value) {
//    // Print does not return any value and has a single argument
//    state->operands.push_back(value);
//}
//
//mlir::LogicalResult PrintOp::verify() {
//    if (failed(verifyRiseSingleOperand(this)))
//        return mlir::failure();
//    return mlir::success();
//}
//
//void TransposeOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                        mlir::Value *value) {
//    state->types.push_back(RiseArrayType::get(builder->getContext()));
//    state->operands.push_back(value);
//}
//
//mlir::LogicalResult TransposeOp::verify() {
//    if (failed(verifyRiseSingleOperand(this)))
//        return mlir::failure();
//    return mlir::success();
//}
//
//void ReshapeOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                      mlir::Value *value, RiseArrayType reshapedType) {
//    state->types.push_back(reshapedType);
//    state->operands.push_back(value);
//}
//
//mlir::LogicalResult ReshapeOp::verify() {
//    if (failed(verifyRiseSingleOperand(this)))
//        return mlir::failure();
//    auto retTy = getResult()->getType().dyn_cast<RiseArrayType>();
//    if (!retTy)
//        return emitOpError("rise.reshape is expected to produce a Rise array");
//    if (retTy.isGeneric())
//        return emitOpError("rise.reshape is expected to produce a shaped Rise array, "
//                           "got a generic one.");
//    return mlir::success();
//}
//
//void AddOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                  mlir::Value *lhs, mlir::Value *rhs) {
//    state->types.push_back(RiseArrayType::get(builder->getContext()));
//    state->operands.push_back(lhs);
//    state->operands.push_back(rhs);
//}
//
//mlir::LogicalResult AddOp::verify() {
//    if (failed(verifyRiseBinOperands(this)))
//        return mlir::failure();
//    return mlir::success();
//}
//
//void MulOp::build(mlir::Builder *builder, mlir::OperationState *state,
//                  mlir::Value *lhs, mlir::Value *rhs) {
//    state->types.push_back(RiseArrayType::get(builder->getContext()));
//    state->operands.push_back(lhs);
//    state->operands.push_back(rhs);
//}
//
//mlir::LogicalResult MulOp::verify() {
//    if (failed(verifyRiseBinOperands(this)))
//        return mlir::failure();
//    return mlir::success();
//}


#define GET_OP_CLASSES
#include "mlir/Dialect/Rise/Ops.cpp.inc"
} //end namespace rise
} //end namespace mlir


