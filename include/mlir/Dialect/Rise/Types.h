//
// Created by martin on 2019-09-23.
//

#ifndef LLVM_TYPES_H
#define LLVM_TYPES_H


#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Rise Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////
namespace mlir {
namespace rise {

namespace detail {
struct ArrayTypeStorage;
struct RiseFunTypeStorage;
struct RiseDataTypeWrapperStorage;
struct RiseTupleTypeStorage;
}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum RiseTypeKind {
    // The enum starts at the range reserved for this dialect.
            RISE_TYPE = mlir::Type::FIRST_RISE_TYPE,
            RISE_BASETYPE,
    RISE_FUNTYPE,
    RISE_DATATYPE_WRAPPER,
    RISE_NAT,
    RISE_NAT_WRAPPER,
    RISE_DATATYPE,
    RISE_INT,
    RISE_FLOAT,
    RISE_TUPLE,
    RISE_ARRAY,
};


class RiseType : public Type::TypeBase<RiseType, Type> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_TYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static RiseType get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, RiseTypeKind::RISE_TYPE);
    }
};


class FunType : public Type::TypeBase<FunType, RiseType, detail::RiseFunTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_FUNTYPE; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_TYPE; }

    static FunType get(mlir::MLIRContext *context,
                       RiseType input, RiseType output);

    static FunType getChecked(mlir::MLIRContext *context, RiseType input, RiseType output,
                                 mlir::Location location);


    static mlir::LogicalResult verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                            mlir::MLIRContext *context,
                                                            RiseType input, RiseType output);

    RiseType getInput();

    RiseType getOutput();
};


class DataType : public Type::TypeBase<DataType, Type> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static DataType get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, RiseTypeKind::RISE_DATATYPE);
    }
};

class DataTypeWrapper : public Type::TypeBase<DataTypeWrapper, RiseType, detail::RiseDataTypeWrapperStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE_WRAPPER; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_TYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static DataTypeWrapper get(mlir::MLIRContext *context, DataType data);
    DataType getData();
};

class Int : public Type::TypeBase<Int, DataType> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_INT; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static Int get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, RiseTypeKind::RISE_INT);
    }
};

class Float : public Type::TypeBase<Float, DataType> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_FLOAT; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static Float get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, RiseTypeKind::RISE_FLOAT);
    }
};

class Tuple : public mlir::Type::TypeBase<Tuple, DataType, detail::RiseTupleTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_TUPLE; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static Tuple get(mlir::MLIRContext *context, DataType first, DataType second);
    DataType getFirst();
    DataType getSecond();
};

/// Type for Rise arrays.
/// In MLIR Types are reference to immutable and uniqued objects owned by the
/// MLIRContext. As such `RiseArrayType` only wraps a pointer to an uniqued
/// instance of `RiseArrayTypeStorage` (defined in our implementation file) and
/// provides the public facade API to interact with the type.
class ArrayType : public mlir::Type::TypeBase<ArrayType, DataType,
        detail::ArrayTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    static mlir::LogicalResult verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                            mlir::MLIRContext *context,
                                                            int size, DataType elementType);

    /// Get the unique instance of this Type from the context.
    static ArrayType get(mlir::MLIRContext *context,
                             int size, DataType elementType);

    /// Support method to enable LLVM-style RTTI type casting.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_ARRAY; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }

    int getSize();
    /// Return the type of individual elements in the array.
    DataType getElementType();

};

class Nat : public mlir::Type::TypeBase<Nat, Type> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_NAT; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_NAT; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static Nat get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, RiseTypeKind::RISE_NAT);
    }
};


class NatTypeWrapper : public mlir::Type::TypeBase<NatTypeWrapper, mlir::Type> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_NAT_WRAPPER; }
    static bool basetype(unsigned kind) { return kind == RiseTypeKind::RISE_DATATYPE; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static NatTypeWrapper get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, RiseTypeKind::RISE_NAT_WRAPPER);
    }
};

} //end namespace rise
} //end namespace mlir

#endif //LLVM_TYPES_H
