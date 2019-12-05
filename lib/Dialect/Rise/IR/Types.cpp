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

FunType FunType::get(mlir::MLIRContext *context, RiseType input, RiseType output) {
    return Base::get(context, RiseTypeKind::RISE_FUNTYPE, input, output);
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

mlir::LogicalResult ArrayType::verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                             mlir::MLIRContext *context,
                                                             int size, DataType elementType) {
    ///For some reason this method is called without a valid location in StorageUniquerSupport

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