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
namespace lift {

namespace detail {
struct LiftTypeAttributeStorage;
}


/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum LiftAttributeKind {
    // The enum starts at the range reserved for this dialect.
    //TODO: FIRST_LIFT_ATTR does not exist
            LIFT_ATTR = mlir::Type::FIRST_LIFT_TYPE,
    LIFT_TYPE_ATTR,
};

class LiftTypeAttr : public Attribute::AttrBase<LiftTypeAttr, Attribute, detail::LiftTypeAttributeStorage> {
public:
    using Base::Base;
//    using ValueType = Type;

    static LiftTypeAttr get(Type value);

    Type getValue() const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == LiftAttributeKind::LIFT_TYPE_ATTR; }
};

}
}
#endif //LLVM_ATTRIBUTES_H
