//
// Created by martin on 2019-09-23.
//
#include "mlir/Dialect/Lift/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"


////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////
using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace lift {


/// Helper to verify that the result of an operation is a Lift array type.
template <typename T> static mlir::LogicalResult verifyLiftReturnArray(T *op) {
    if (!op->getResult()->getType().template isa<LiftArrayType>()) {
        std::string msg;
        raw_string_ostream os(msg);
        os << "expects a Lift Array for its argument, got "
           << op->getResult()->getType();
        return op->emitOpError(os.str());
    }
    return mlir::success();
}

/// Helper to verify that the two operands of a binary operation are Lift
/// arrays..
template <typename T> static mlir::LogicalResult verifyLiftBinOperands(T *op) {
    if (!op->getOperand(0)->getType().template isa<LiftArrayType>()) {
        std::string msg;
        raw_string_ostream os(msg);
        os << "expects a Lift Array for its LHS, got "
           << op->getOperand(0)->getType();
        return op->emitOpError(os.str());
    }
    if (!op->getOperand(1)->getType().template isa<LiftArrayType>()) {
        std::string msg;
        raw_string_ostream os(msg);
        os << "expects a Lift Array for its LHS, got "
           << op->getOperand(0)->getType();
        return op->emitOpError(os.str());
    }
    return mlir::success();
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::Builder *builder, mlir::OperationState *state,
                       ArrayRef<int64_t> shape, mlir::DenseElementsAttr value) {
    state->types.push_back(LiftArrayType::get(builder->getContext(), shape));
    auto dataAttribute = builder->getNamedAttr("value", value);
    state->attributes.push_back(dataAttribute);
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::Builder *builder, mlir::OperationState *state,
                       mlir::FloatAttr value) {
    // Broadcast and forward to the other build factory
    mlir::Type elementType = mlir::FloatType::getF64(builder->getContext());
    auto dataType = builder->getTensorType({1}, elementType);
    auto dataAttribute = builder->getDenseElementsAttr(dataType, {value})
            .cast<mlir::DenseElementsAttr>();

    ConstantOp::build(builder, state, {1}, dataAttribute);
}

/// Verifier for constant operation.
mlir::LogicalResult ConstantOp::verify() {
    // Ensure that the return type is a Lift array
    if (failed(verifyLiftReturnArray(this)))
        return mlir::failure();

    // We expect the constant itself to be stored as an attribute.
    auto dataAttr = getAttr("value").dyn_cast<mlir::DenseElementsAttr>();
    if (!dataAttr) {
        return emitOpError(
                "missing valid `value` DenseElementsAttribute on lift.constant()");
    }
    auto attrType = dataAttr.getType().dyn_cast<mlir::TensorType>();
    if (!attrType) {
        return emitOpError(
                "missing valid `value` DenseElementsAttribute on lift.constant()");
    }

    // If the return type of the constant is not a generic array, the shape must
    // match the shape of the attribute holding the data.
    auto resultType = getResult()->getType().cast<LiftArrayType>();
    if (!resultType.isGeneric()) {
        if (attrType.getRank() != resultType.getRank()) {
            return emitOpError("The rank of the lift.constant return type must match "
                               "the one of the attached value attribute: " +
                               Twine(attrType.getRank()) +
                               " != " + Twine(resultType.getRank()));
        }
        for (int dim = 0; dim < attrType.getRank(); ++dim) {
            if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
                std::string msg;
                raw_string_ostream os(msg);
                return emitOpError(
                        "Shape mismatch between lift.constant return type and its "
                        "attribute at dimension " +
                        Twine(dim) + ": " + Twine(attrType.getShape()[dim]) +
                        " != " + Twine(resultType.getShape()[dim]));
            }
        }
    }
    return mlir::success();
}

void GenericCallOp::build(mlir::Builder *builder, mlir::OperationState *state,
                          StringRef callee, ArrayRef<mlir::Value *> arguments) {
    // Generic call always returns a generic LiftArray initially
    state->types.push_back(LiftArrayType::get(builder->getContext()));
    state->operands.assign(arguments.begin(), arguments.end());
    auto calleeAttr = builder->getStringAttr(callee);
    state->attributes.push_back(builder->getNamedAttr("callee", calleeAttr));
}

mlir::LogicalResult GenericCallOp::verify() {
    // Verify that every operand is a Lift Array
    for (int opId = 0, num = getNumOperands(); opId < num; ++opId) {
        if (!getOperand(opId)->getType().template isa<LiftArrayType>()) {
            std::string msg;
            raw_string_ostream os(msg);
            os << "expects a Lift Array for its " << opId << " operand, got "
               << getOperand(opId)->getType();
            return emitOpError(os.str());
        }
    }
    return mlir::success();
}

/// Return the name of the callee.
StringRef GenericCallOp::getCalleeName() {
    return getAttr("callee").cast<mlir::StringAttr>().getValue();
}

template <typename T> static mlir::LogicalResult verifyLiftSingleOperand(T *op) {
    if (!op->getOperand()->getType().template isa<LiftArrayType>()) {
        std::string msg;
        raw_string_ostream os(msg);
        os << "expects a Lift Array for its argument, got "
           << op->getOperand()->getType();
        return op->emitOpError(os.str());
    }
    return mlir::success();
}

void ReturnOp::build(mlir::Builder *builder, mlir::OperationState *state,
                     mlir::Value *value) {
    // Return does not return any value and has an optional single argument
    if (value)
        state->operands.push_back(value);
}

mlir::LogicalResult ReturnOp::verify() {
    if (getNumOperands() > 1) {
        std::string msg;
        raw_string_ostream os(msg);
        os << "expects zero or one operand, got " << getNumOperands();
        return emitOpError(os.str());
    }
    if (hasOperand() && failed(verifyLiftSingleOperand(this)))
        return mlir::failure();
    return mlir::success();
}

void PrintOp::build(mlir::Builder *builder, mlir::OperationState *state,
                    mlir::Value *value) {
    // Print does not return any value and has a single argument
    state->operands.push_back(value);
}

mlir::LogicalResult PrintOp::verify() {
    if (failed(verifyLiftSingleOperand(this)))
        return mlir::failure();
    return mlir::success();
}

void TransposeOp::build(mlir::Builder *builder, mlir::OperationState *state,
                        mlir::Value *value) {
    state->types.push_back(LiftArrayType::get(builder->getContext()));
    state->operands.push_back(value);
}

mlir::LogicalResult TransposeOp::verify() {
    if (failed(verifyLiftSingleOperand(this)))
        return mlir::failure();
    return mlir::success();
}

void ReshapeOp::build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *value, LiftArrayType reshapedType) {
    state->types.push_back(reshapedType);
    state->operands.push_back(value);
}

mlir::LogicalResult ReshapeOp::verify() {
    if (failed(verifyLiftSingleOperand(this)))
        return mlir::failure();
    auto retTy = getResult()->getType().dyn_cast<LiftArrayType>();
    if (!retTy)
        return emitOpError("lift.reshape is expected to produce a Lift array");
    if (retTy.isGeneric())
        return emitOpError("lift.reshape is expected to produce a shaped Lift array, "
                           "got a generic one.");
    return mlir::success();
}

void AddOp::build(mlir::Builder *builder, mlir::OperationState *state,
                  mlir::Value *lhs, mlir::Value *rhs) {
    state->types.push_back(LiftArrayType::get(builder->getContext()));
    state->operands.push_back(lhs);
    state->operands.push_back(rhs);
}

mlir::LogicalResult AddOp::verify() {
    if (failed(verifyLiftBinOperands(this)))
        return mlir::failure();
    return mlir::success();
}

void MulOp::build(mlir::Builder *builder, mlir::OperationState *state,
                  mlir::Value *lhs, mlir::Value *rhs) {
    state->types.push_back(LiftArrayType::get(builder->getContext()));
    state->operands.push_back(lhs);
    state->operands.push_back(rhs);
}

mlir::LogicalResult MulOp::verify() {
    if (failed(verifyLiftBinOperands(this)))
        return mlir::failure();
    return mlir::success();
}

} //end namespace lift