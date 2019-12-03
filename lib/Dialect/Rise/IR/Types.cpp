//
// Created by martin on 2019-09-23.
//

#include <iostream>
#include "mlir/Dialect/Rise/Types.h"
#include "mlir/Dialect/Rise/TypeDetail.h"
#include "mlir/Dialect/Rise/Dialect.h"

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
namespace rise {



//===----------------------------------------------------------------------===//
// DataTypeWrapper
//===----------------------------------------------------------------------===//

DataType DataTypeWrapper::getData() {
    return getImpl()->data;
}

DataTypeWrapper DataTypeWrapper::get(mlir::MLIRContext *context, DataType data) {
    return Base::get(context, RiseTypeKind::RISE_DATATYPE_WRAPPER, data);
}

//===----------------------------------------------------------------------===//
// FunType
//===----------------------------------------------------------------------===//

/// This method is used to get an instance of the 'ComplexType'. This method
/// asserts that all of the construction invariants were satisfied. To
/// gracefully handle failed construction, getChecked should be used instead.
FunType FunType::get(mlir::MLIRContext *context, RiseType input, RiseType output) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // type kind.
    return Base::get(context, RiseTypeKind::RISE_FUNTYPE, input, output);
}

/// This method is used to get an instance of the 'ComplexType', defined at
/// the given location. If any of the construction invariants are invalid,
/// errors are emitted with the provided location and a null type is returned.
/// Note: This method is completely optional
FunType FunType::getChecked(mlir::MLIRContext *context, RiseType input, RiseType output,
                                  mlir::Location location) {
    return Base::getChecked(location, context, RiseTypeKind::RISE_FUNTYPE, input, output);
}

/// This method is used to verify the construction invariants passed into the
/// 'get' and 'getChecked' methods. Note: This method is completely optional.
mlir::LogicalResult FunType::verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                             mlir::MLIRContext *context,
                                                          RiseType input, RiseType output) {
    ///For some reason this method is called without a valid location in StorageUniquerSupport



    ///verifying with .isa is a NoNo, because we might have a sublclass of RiseType
//    if (!input.isa<RiseType>()) {
//        if (loc) {
//            emitError(loc.getValue(), "input is not a valid Type to construct LambdaType");
//        } else {
//            emitError(UnknownLoc::get(context), "input is not a valid Type to construct LambdaType");
//        }
//        return mlir::failure();
//    }
//    if (!output.isa<RiseType>()) {
//        if (loc) {
//            emitError(loc.getValue(), "output is not a valid Type to construct LambdaType");
//        } else {
//            emitError(UnknownLoc::get(context), "output is not a valid Type to construct LambdaType");
//        }
//        return mlir::failure();
//    }
    return mlir::success();
}

RiseType FunType::getInput() {
    return getImpl()->input;
}

RiseType FunType::getOutput() {
    return getImpl()->output;
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//
Tuple rise::Tuple::get(mlir::MLIRContext *context, DataType first, DataType second) {
    return Base::get(context, RiseTypeKind::RISE_TUPLE, first, second);
}

DataType rise::Tuple::getFirst() {
    return getImpl()->getFirst();
}
DataType rise::Tuple::getSecond() {
    return getImpl()->getSecond();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

int ArrayType::getSize() { return getImpl()->getSize(); }

DataType ArrayType::getElementType() {
    return getImpl()->getElementType();
}

ArrayType ArrayType::get(mlir::MLIRContext *context,
                                 int size, DataType elementType) {
    return Base::get(context, RiseTypeKind::RISE_ARRAY, size, elementType);
}
/// This method is used to verify the construction invariants passed into the
/// 'get' and 'getChecked' methods. Note: This method is completely optional.
mlir::LogicalResult ArrayType::verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                             mlir::MLIRContext *context,
                                                             int size, DataType elementType) {
    ///For some reason this method is called without a valid location in StorageUniquerSupport

    //TODO: Determine here all valid types which an Array can contain

    if (!(size > 0)) {
        if (loc) {
            emitError(loc.getValue(), "Arrays have to contain at least 1 element");
        } else {
            emitError(UnknownLoc::get(context), "Arrays have to contain at least 1 element");
        }
        return mlir::failure();
    }


    return mlir::success();
}

} //end namespace rise
} //end namespace mlir