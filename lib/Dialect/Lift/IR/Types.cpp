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