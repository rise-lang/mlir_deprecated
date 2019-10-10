//#include "mlir/IR/Types.h"

// Created by martin on 10/10/2019.
//

#include "mlir/Dialect/Lift/Attributes.h"
#include "mlir/Dialect/Lift/AttributeDetail.h"
#include "mlir/Dialect/Lift/Dialect.h"

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
namespace lift {



//===----------------------------------------------------------------------===//
// LiftTypeAttr
//===----------------------------------------------------------------------===//

LiftTypeAttr LiftTypeAttr::get(Type value) {
    return Base::get(value.getContext(), LiftAttributeKind::LIFT_TYPE_ATTR, value);
}

Type LiftTypeAttr::getValue() const { return getImpl()->value; }



}
}