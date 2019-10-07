//
// Created by martin on 2019-09-23.
//

#ifndef LLVM_OPS_H
#define LLVM_OPS_H

#include "Types.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Constant operation turns a literal into an SSA value. The data is attached
/// to the operation as an attribute. For example:
///
///   %0 = "lift.constant"()
///       {value: dense<tensor<2x3xf64>, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>}
///     : () -> !lift.array<2, 3>
///
/// An operation inherits from `class Op` and specifies optional traits. Here we
/// indicate that `lift.constant` does not have any operands and returns a single
/// result. The traits provide some utilities methods for the operation, for
/// instance we will be able to use `getResult()`, but `getOperand()` won't be
/// available.
namespace  mlir {
namespace lift {


#define GET_OP_CLASSES

#include "mlir/Dialect/Lift/Ops.h.inc"


class ConstantOp : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
        mlir::OpTrait::OneResult,
        mlir::OpTrait::HasNoSideEffect> {
public:
    /// This is the name used by MLIR to match an operation to this class during
    /// parsing.
    static llvm::StringRef getOperationName() { return "lift.constant"; }

    /// The operation can have extra verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to mlir::Builder::create<PrintOp>(...)
    /// This method populates the `state` that MLIR uses to create operations.
    /// The `lift.constant` operation does not have arguments but attaches a
    /// constant array as an attribute and returns it as an SSA value.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      llvm::ArrayRef<int64_t> shape,
                      mlir::DenseElementsAttr value);

    /// Similar to the one above, but takes a single float and returns a
    /// !lift.array<1>.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::FloatAttr value);

    /// Inherit constructor.
    using Op::Op;
};

/// Generic calls represent calls to a user defined function that needs to
/// be specialized for the shape of its arguments. The callee name is attached
/// as a literal string as an attribute. The arguments list must match the
/// arguments expected by the callee. For example:
///
///   %4 = "lift.generic_call"(%1, %3) {callee: "my_func"}
///         : (!lift.array<2, 3>, !lift.array<2, 3>) -> !lift<"array">
///
/// This is only valid if a function named "my_func" exists and takes two
/// arguments.
class GenericCallOp
        : public mlir::Op<GenericCallOp, mlir::OpTrait::VariadicOperands,
                mlir::OpTrait::OneResult> {
public:
    /// MLIR will use this to register the operation with the parser/printer.
    static llvm::StringRef getOperationName() { return "lift.generic_call"; }

    /// Operations can add custom verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to the builder to allow:
    ///   mlir::Builder::create<GenericCallOp>(...)
    /// This method populate the `state` that MLIR use to create operations.
    /// The `lift.generic_call` operation accepts a callee name and a list of
    /// arguments for the call.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      llvm::StringRef callee,
                      llvm::ArrayRef<mlir::Value *> arguments);

    /// Return the name of the callee.
    llvm::StringRef getCalleeName();

    /// Inherit constructor.
    using Op::Op;
};


/// Generic calls represent calls to a user defined function that needs to
/// be specialized for the shape of its arguments. The callee name is attached
/// as a literal string as an attribute. The arguments list must match the
/// arguments expected by the callee. For example:
///
///   %4 = "lift.generic_call"(%1, %3) {callee: "my_func"}
///         : (!lift.array<2, 3>, !lift.array<2, 3>) -> !lift<"array">
///
/// This is only valid if a function named "my_func" exists and takes two
/// arguments.
//class ApplyOp
//: public mlir::Op<ApplyOp, mlir::OpTrait::NOperands<2>::Impl,
//                mlir::OpTrait::OneResult> {
//public:
//    /// MLIR will use this to register the operation with the parser/printer.
//    static llvm::StringRef getOperationName() { return "lift.apply"; }
//
//    /// Operations can add custom verification beyond the traits they define.
//    mlir::LogicalResult verify();
//
//    /// Interface to the builder to allow:
//    ///   mlir::Builder::create<GenericCallOp>(...)
//    /// This method populate the `state` that MLIR use to create operations.
//    /// The `lift.generic_call` operation accepts a callee name and a list of
//    /// arguments for the call.
//    static void build(mlir::Builder *builder, mlir::OperationState *state,
//                      llvm::StringRef callee,
//                      llvm::ArrayRef<mlir::Value *> arguments);
//
//    /// Return the name of the callee.
//    llvm::StringRef getCalleeName();
//
//    /// Inherit constructor.
//    using Op::Op;
//};




///// Return operations terminate blocks (and functions as well). They take a
///// single argument and the type must match the function return type.
//class ReturnOp
//        : public mlir::Op<ReturnOp, mlir::OpTrait::VariadicOperands,
//                mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
//public:
//    static llvm::StringRef getOperationName() { return "lift.return"; }
//
//    /// Operations can add custom verification beyond the traits they define.
//    mlir::LogicalResult verify();
//
//    /// Interface to mlir::Builder::create<PrintOp>(...)
//    /// This method populate the `state` that MLIR use to create operations.
//    /// The `lift.return` operation accepts an optional single array as an argument
//    /// and does not have any returned value.
//    static void build(mlir::Builder *builder, mlir::OperationState *state,
//                      mlir::Value *value = nullptr);
//
//    /// Return true if there is a returned value.
//    bool hasOperand() { return 0 != getNumOperands(); }
//
//    /// Helper to return the optional operand. Caller must check if the operand
//    /// is present before calling this.
//    mlir::Value *getOperand() { return getOperation()->getOperand(0); }
//
//    /// Inherit constructor.
//    using Op::Op;
//};

/// The print builtin takes a single array argument and does not return any.
class PrintOp : public mlir::Op<PrintOp, mlir::OpTrait::OneOperand,
        mlir::OpTrait::ZeroResult> {
public:
    static llvm::StringRef getOperationName() { return "lift.print"; }

    /// Operations can add custom verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to mlir::Builder::create<PrintOp>(...)
    /// This method populate the `state` that MLIR use to create operations.
    /// The `lift.print` operation accepts a single array as argument and does
    /// not have any returned value.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *value);

    /// Inherit constructor.
    using Op::Op;
};

class TransposeOp : public mlir::Op<TransposeOp, mlir::OpTrait::OneOperand,
        mlir::OpTrait::OneResult> {
public:
    static llvm::StringRef getOperationName() { return "lift.transpose"; }

    /// Operation can add custom verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to mlir::Builder::create<TransposeOp>(...)
    /// This method populate the `state` that MLIR use to create operations.
    /// The `lift.transpose` operation accepts a single array as argument and
    /// returns the transposed array as its only result.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *value);

    /// Inherit constructor.
    using Op::Op;
};

/// Reshape operation is transforming its input array into a new array with the
/// same number of elements but different shapes. For example:
///
///    %0 = "lift.reshape"(%arg1) : (!lift.array<10>) -> !lift.array<5, 2>
///
class ReshapeOp : public mlir::Op<ReshapeOp, mlir::OpTrait::OneOperand,
        mlir::OpTrait::OneResult> {
public:
    static llvm::StringRef getOperationName() { return "lift.reshape"; }

    /// Operation can add custom verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to mlir::Builder::create<ReshapeOp>(...)
    /// This method populate the `state` that MLIR use to create operations.
    /// The `lift.reshape` operation accepts a single array as argument and
    /// returns the array with the specified reshapedType as its only result.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *value, LiftArrayType reshapedType);

    /// Inherit constructor.
    using Op::Op;
};

/// Binary operation implementing a multiplication. For two-dimensional array
/// a matrix multiplication is implemented, while for one dimensional array a
/// dot product is performed.
class MulOp : public mlir::Op<MulOp, mlir::OpTrait::NOperands<2>::Impl,
        mlir::OpTrait::OneResult> {
public:
    static llvm::StringRef getOperationName() { return "lift.mul"; }

    /// Operation can add custom verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to mlir::Builder::create<PrintOp>(...)
    /// This method populate the `state` that MLIR use to create operations.
    /// The `lift.mul` operation accepts two operands as argument and returns
    /// a single value.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *lhs, mlir::Value *rhs);

    /// Convenience accessor for LHS of the expression.
    mlir::Value *getLHS() { return getOperand(0); }

    /// Convenience accessor for RHS of the expression.
    mlir::Value *getRHS() { return getOperand(1); }

    /// Inherit constructor.
    using Op::Op;
};

/// Element wise addition of two arrays. The shape must match.
class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
        mlir::OpTrait::OneResult> {
public:
    static llvm::StringRef getOperationName() { return "lift.add"; }

    /// Operation can add custom verification beyond the traits they define.
    mlir::LogicalResult verify();

    /// Interface to mlir::Builder::create<PrintOp>(...)
    /// This method populate the `state` that MLIR use to create operations.
    /// The `lift.mul` operation accepts two operands as argument and returns
    /// a single value.
    static void build(mlir::Builder *builder, mlir::OperationState *state,
                      mlir::Value *lhs, mlir::Value *rhs);

    /// Convenience accessor for LHS of the expression.
    mlir::Value *getLHS() { return getOperand(0); }

    /// Convenience accessor for RHS of the expression.
    mlir::Value *getRHS() { return getOperand(1); }

    /// Inherit constructor.
    using Op::Op;
};

} //end namespace lift
} //end namespace mlir

#endif //LLVM_OPS_H
