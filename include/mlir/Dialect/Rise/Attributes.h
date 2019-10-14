//
// Created by martin on 10/10/2019.
//

#ifndef LLVM_ATTRIBUTES_H
#define LLVM_ATTRIBUTES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/AttributeSupport.h"
#include "AttributeDetail.h"
#include "mlir/IR/Attributes.h"

namespace mlir {
namespace rise {

namespace detail {
struct RiseTypeAttributeStorage;
}


/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum RiseAttributeKind {
    // The enum starts at the range reserved for this dialect.
    //TODO: FIRST_RISE_ATTR does not exist
            RISE_ATTR = mlir::Type::FIRST_RISE_TYPE,
    RISE_TYPE_ATTR,
};

class RiseTypeAttr : public Attribute::AttrBase<RiseTypeAttr, Attribute, detail::RiseTypeAttributeStorage> {
public:
    using Base::Base;
//    using ValueType = Type;

    static RiseTypeAttr get(mlir::Type value);

    mlir::Type getValue() const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseAttributeKind::RISE_TYPE_ATTR; }
};

}
}
#endif //LLVM_ATTRIBUTES_H
