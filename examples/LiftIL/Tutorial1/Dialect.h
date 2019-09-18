//
// Created by martin on 2019-09-18.
//

#ifndef MLIR_DIALECT_H
#define MLIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"


class LiftDialect : public mlir::Dialect {
public:
    explicit LiftDialect(MLIRContext *ctx);


    /// Parse a type registered to this dialect. Overridding this method is
    /// required for dialects that have custom types.
    /// Technically this is only needed to be able to round-trip to textual IR.
    mlir::Type parseType(llvm::StringRef tyData,
                         mlir::Location loc) const override;

    /// Print a type registered to this dialect. Overridding this method is
    /// only required for dialects that have custom types.
    /// Technically this is only needed to be able to round-trip to textual IR.
    void printType(mlir::Type type, llvm::raw_ostream &os) const override;
};

    ///
    /// Custom Types
    ///
namespace detail{
    struct LiftArrayTypeStorage;
}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum LiftTypeKind {
    // The enum starts at the range reserved for this dialect.
    LIFT_TYPE = mlir::Type::FIRST_LIFT_TYPE,
    LIFT_ARRAY,
};


class LiftArrayType : public mlir::Type::Base<LiftArrayType, mlir::Type, detail::LiftArrayTypeStorage> {

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


#endif //MLIR_DIALECT_H
