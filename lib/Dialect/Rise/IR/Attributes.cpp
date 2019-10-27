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
// LiteralAttr
//===----------------------------------------------------------------------===//

LiteralAttr LiteralAttr::get(MLIRContext *context, DataType type, std::string value) {
    return Base::get(context, RiseAttributeKind::LITERAL_ATTR, type, value);
}
DataType LiteralAttr::getType() const { return getImpl()->type; }

std::string LiteralAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// RiseTypeAttr
//===----------------------------------------------------------------------===//

RiseTypeAttr RiseTypeAttr::get(MLIRContext *context, RiseType value) {
    return Base::get(context, RiseAttributeKind::RISETYPE_ATTR, value);
}
RiseType RiseTypeAttr::getValue() const { return getImpl()->value; }



//===----------------------------------------------------------------------===//
// DataTypeAttr
//===----------------------------------------------------------------------===//

DataTypeAttr DataTypeAttr::get(MLIRContext *context, DataType value) {
    return Base::get(context, RiseAttributeKind::DATATYPE_ATTR, value);
}
DataType DataTypeAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// NatAttr
//===----------------------------------------------------------------------===//

NatAttr NatAttr::get(MLIRContext *context, int value) {
    return Base::get(context, RiseAttributeKind::NAT_ATTR, value);
}
int NatAttr::getValue() const { return getImpl()->value; }



}
}