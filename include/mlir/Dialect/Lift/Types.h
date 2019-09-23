//
// Created by martin on 2019-09-23.
//

#ifndef LLVM_TYPES_H
#define LLVM_TYPES_H


#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Lift Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace lift {

namespace detail {
struct LiftArrayTypeStorage;
}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum LiftTypeKind {
    // The enum starts at the range reserved for this dialect.
            LIFT_KIND = mlir::Type::FIRST_LIFT_TYPE,
    LIFT_NAT,
    LIFT_DATA,
    LIFT_ARRAY,
};

class Kind : public mlir::Type::TypeBase<Kind, mlir::Type> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == LiftTypeKind::LIFT_KIND; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static Kind get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, LiftTypeKind::LIFT_KIND);
    }
};

class Nat : public mlir::Type::TypeBase<Nat, Kind> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == LiftTypeKind::LIFT_NAT; }

    /// This method is used to get an instance of the 'SimpleType'. Given that
    /// this is a parameterless type, it just needs to take the context for
    /// uniquing purposes.
    static Kind get(mlir::MLIRContext *context) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type.
        return Base::get(context, LiftTypeKind::LIFT_NAT);
    }
};








/// Type for Lift arrays.
/// In MLIR Types are reference to immutable and uniqued objects owned by the
/// MLIRContext. As such `LiftArrayType` only wraps a pointer to an uniqued
/// instance of `LiftArrayTypeStorage` (defined in our implementation file) and
/// provides the public facade API to interact with the type.
class LiftArrayType : public mlir::Type::TypeBase<LiftArrayType, mlir::Type,
        detail::LiftArrayTypeStorage> {

public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// Returns the dimensions for this array, or and empty range for a generic
    /// array.
    llvm::ArrayRef<int64_t> getShape();

    /// Predicate to test if this array is generic (shape haven't been inferred
    /// yet).
    bool isGeneric() { return getShape().empty(); }

    /// Return the rank of this array (0 if it is generic).
    int getRank() { return getShape().size(); }

    /// Return the type of individual elements in the array.
    mlir::Type getElementType();

    /// Get the unique instance of this Type from the context.
    /// A LiftArrayType is only defined by the shape of the array.
    static LiftArrayType get(mlir::MLIRContext *context,
                             llvm::ArrayRef<int64_t> shape = {});

    /// Support method to enable LLVM-style RTTI type casting.
    static bool kindof(unsigned kind) { return kind == LiftTypeKind::LIFT_ARRAY; }
};

} //end namespace lift

#endif //LLVM_TYPES_H
