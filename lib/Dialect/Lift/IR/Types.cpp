//
// Created by martin on 2019-09-23.
//

#include "mlir/Dialect/Lift/Types.h"
#include "mlir/Dialect/Lift/TypeDetail.h"
#include "mlir/Dialect/Lift/Dialect.h"

#include "mlir/IR/Diagnostics.h"
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

namespace mlir {
namespace lift {









//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

/// This method is used to get an instance of the 'ComplexType'. This method
/// asserts that all of the construction invariants were satisfied. To
/// gracefully handle failed construction, getChecked should be used instead.
LambdaType LambdaType::get(mlir::MLIRContext *context, Type input, Type output) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // type kind.
    return Base::get(context, LiftTypeKind::LIFT_LAMBDA, input, output);
}

/// This method is used to get an instance of the 'ComplexType', defined at
/// the given location. If any of the construction invariants are invalid,
/// errors are emitted with the provided location and a null type is returned.
/// Note: This method is completely optional
LambdaType LambdaType::getChecked(mlir::MLIRContext *context, Type input, Type output,
                                      mlir::Location location) {
    return Base::getChecked(location, context, LiftTypeKind::LIFT_LAMBDA, input, output);
}

/// This method is used to verify the construction invariants passed into the
/// 'get' and 'getChecked' methods. Note: This method is completely optional.
mlir::LogicalResult LambdaType::verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                               mlir::MLIRContext *context,
                                                             Type input, Type output) {
    ///For some reason this method is called without a valid location in StorageUniquerSupport

    //Determine here all valid types for Lambda in and out

    if (!input.isa<Type>()) {
        if (loc) {
            emitError(loc.getValue(), "input is not a valid Type to construct LambdaType");
        } else {
            emitError(UnknownLoc::get(context), "input is not a valid Type to construct LambdaType");
        }
        return mlir::failure();
    }
    if (!output.isa<Type>()) {
        if (loc) {
            emitError(loc.getValue(), "output is not a valid Type to construct LambdaType");
        } else {
            emitError(UnknownLoc::get(context), "output is not a valid Type to construct LambdaType");
        }
        return mlir::failure();
    }
    return mlir::success();
}

Type LambdaType::getInput() {
    return getImpl()->input;
}

Type LambdaType::getOutput() {
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

} //end namespace lift
} //end namespace mlir