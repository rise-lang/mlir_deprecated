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
struct DataTypeAttributeStorage;
struct NatAttributeStorage;
struct LiteralAttributeStorage;
}


/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum RiseAttributeKind {
    // The enum starts at the range reserved for this dialect.
            RISE_ATTR = mlir::Attribute::FIRST_RISE_ATTR,
    NAT_ATTR,
            LITERAL_ATTR,
            RISETYPE_ATTR,
            DATATYPE_ATTR,

};


///Attributes are used to specify constant data on operations
class LiteralAttr : public Attribute::AttrBase<LiteralAttr, Attribute, detail::LiteralAttributeStorage> {
public:
    using Base::Base;

    static LiteralAttr get(MLIRContext *context, DataType type, std::string value);

    std::string getValue() const;
    DataType getType() const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseAttributeKind::LITERAL_ATTR; }
};




class DataTypeAttr : public Attribute::AttrBase<DataTypeAttr, Attribute, detail::DataTypeAttributeStorage> {
public:
    using Base::Base;

    static DataTypeAttr get(MLIRContext *context, DataType value);

//    std::string getValue() const;
    DataType getValue() const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseAttributeKind::DATATYPE_ATTR; }
};

class RiseTypeAttr : public Attribute::AttrBase<RiseTypeAttr, Attribute, detail::RiseTypeAttributeStorage> {
public:
    using Base::Base;

    static RiseTypeAttr get(MLIRContext *context, RiseType value);

    RiseType getValue() const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseAttributeKind::RISETYPE_ATTR; }
};

class NatAttr : public Attribute::AttrBase<NatAttr, Attribute, detail::NatAttributeStorage> {
public:
    using Base::Base;

    static NatAttr get(MLIRContext *context, int value);

    int getValue() const;
//    Nat getType() const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseAttributeKind::NAT_ATTR; }
};



}
}
#endif //LLVM_ATTRIBUTES_H
;