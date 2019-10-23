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

RiseTypeAttr RiseTypeAttr::get(MLIRContext *context, mlir::Type type, std::string value) {
    return Base::get(context, RiseAttributeKind::RISE_TYPE_ATTR, type, value);
}
mlir::Type RiseTypeAttr::getType() const { return getImpl()->type; }

std::string RiseTypeAttr::getValue() const { return getImpl()->value; }

}
}