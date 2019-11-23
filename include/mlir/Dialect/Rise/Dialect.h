//===- Dialect.h - Dialect definition for the Rise IR ----------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the IR Dialect for the Rise language.
// See g3doc/Tutorials/Rise/Ch-3.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_RISE_DIALECT_H_
#define MLIR_TUTORIAL_RISE_DIALECT_H_

#include "mlir/IR/Dialect.h"

#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"

#include "mlir/Dialect/Rise/Types.h"
#include "mlir/Dialect/Rise/Ops.h"



namespace mlir {
class Builder;


namespace rise {

/// This is the definition of the Rise dialect. A dialect inherits from
/// mlir::Dialect and register custom operations and types (in its constructor).
/// It can also overridding general behavior of dialects exposed as virtual
/// method, for example regarding verification and parsing/printing.
class RiseDialect : public mlir::Dialect {
public:
    explicit RiseDialect(mlir::MLIRContext *ctx);

    /// Parse a type registered to this dialect. Overridding this method is
    /// required for dialects that have custom types.
    /// Technically this is only needed to be able to round-trip to textual IR.
    mlir::Type parseType(DialectAsmParser &parser) const override;

    RiseType parseRiseType(StringRef typeString,
            mlir::Location loc) const;

    DataType parseDataType(StringRef typeString,
            mlir::Location loc) const;

    Nat parseNat(StringRef typeString,
            mlir::Location loc) const;

        /// Print a type registered to this dialect. Overridding this method is
    /// only required for dialects that have custom types.
    /// Technically this is only needed to be able to round-trip to textual IR.
    void printType(mlir::Type type, DialectAsmPrinter &) const override;


    mlir::Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

    LiteralAttr parseLiteralAttribute(StringRef attrString,
            mlir::Location loc) const;

    RiseTypeAttr parseRiseTypeAttribute(StringRef attrString,
            mlir::Location loc) const;

    DataTypeAttr parseDataTypeAttribute(StringRef attrString,
            mlir::Location loc) const;

    NatAttr parseNatAttribute(StringRef attrString,
            mlir::Location loc) const;
    /// Print an attribute registered to this dialect. Note: The type of the
    /// attribute need not be printed by this method as it is always printed by
    /// the caller.
    void printAttribute(Attribute, DialectAsmPrinter &) const override;
};

} //end namespace rise
} //end namespace mlir
#endif // MLIR_TUTORIAL_RISE_DIALECT_H_
