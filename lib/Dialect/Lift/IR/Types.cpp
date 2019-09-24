//
// Created by martin on 2019-09-23.
//

#include "mlir/Dialect/Lift/Types.h"
#include "mlir/Dialect/Lift/TypeDetail.h"
#include "mlir/Dialect/Lift/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace lift {









//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

/// This method is used to get an instance of the 'ComplexType'. This method
/// asserts that all of the construction invariants were satisfied. To
/// gracefully handle failed construction, getChecked should be used instead.
FunctionType FunctionType::get(mlir::MLIRContext *context, FunctionType input, FunctionType output) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // type kind.
    return Base::get(context, LiftTypeKind::LIFT_FUNCTIONTYPE, input, output);
}

/// This method is used to get an instance of the 'ComplexType', defined at
/// the given location. If any of the construction invariants are invalid,
/// errors are emitted with the provided location and a null type is returned.
/// Note: This method is completely optional
FunctionType FunctionType::getChecked(mlir::MLIRContext *context, FunctionType input, FunctionType output,
                                      mlir::Location location) {
    return Base::getChecked(location, context, LiftTypeKind::LIFT_FUNCTIONTYPE, input, output);
}

/// This method is used to verify the construction invariants passed into the
/// 'get' and 'getChecked' methods. Note: This method is completely optional.
mlir::LogicalResult FunctionType::verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                               mlir::MLIRContext *context,
                                                               FunctionType input, FunctionType output) {
    if (!input.isa<FunctionType>()) {
//            if (loc)
//                context->emitError(loc) << "input is not a valid Type to construct FunctionType";
        return mlir::failure();
    }
    if (!output.isa<FunctionType>()) {
        ///For some reason emitError is not defined.
//            if (loc)
//                context->emitError(loc) << "output is not a valid Type to construct FunctionType";
//                  looks like the call to context is not needed, but then we ant use llvm::Optional anymore. Look into this
        return mlir::failure();
    }
    return mlir::success();
}

FunctionType FunctionType::getInput() {
    return getImpl()->input;
}
FunctionType FunctionType::getOutput() {
    return getImpl()->output;
}

//===----------------------------------------------------------------------===//
// LiftArrayType
//===----------------------------------------------------------------------===//

mlir::Type LiftArrayType::getElementType() {
    return mlir::FloatType::getF64(getContext());
}

LiftArrayType LiftArrayType::get(mlir::MLIRContext *context,
                                 ArrayRef<int64_t> shape) {
    return Base::get(context, LiftTypeKind::LIFT_ARRAY, shape);
}

ArrayRef<int64_t> LiftArrayType::getShape() { return getImpl()->getShape(); }

} //namespace lift