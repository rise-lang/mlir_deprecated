//#include "mlir/IR/Types.h"

// Created by martin on 10/10/2019.
//

#include "mlir/Dialect/Rise/Attributes.h"
#include "mlir/Dialect/Rise/AttributeDetail.h"
#include "mlir/Dialect/Rise/Dialect.h"

#include "mlir/IR/Attributes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir {
namespace rise {



//===----------------------------------------------------------------------===//
// RiseTypeAttr
//===----------------------------------------------------------------------===//

RiseTypeAttr RiseTypeAttr::get(mlir::Type value) {
    return Base::get(value.getContext(), RiseAttributeKind::RISE_TYPE_ATTR, value);
}

mlir::Type RiseTypeAttr::getValue() const { return getImpl()->value; }



}
}