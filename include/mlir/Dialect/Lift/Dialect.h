//===- Dialect.h - Dialect definition for the Lift IR ----------------------===//
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
// This file implements the IR Dialect for the Lift language.
// See g3doc/Tutorials/Lift/Ch-3.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_LIFT_DIALECT_H_
#define MLIR_TUTORIAL_LIFT_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "mlir/Dialect/Lift/Types.h"
#include "mlir/Dialect/Lift/Ops.h"



namespace mlir {
class Builder;
}

namespace lift {

/// This is the definition of the Lift dialect. A dialect inherits from
/// mlir::Dialect and register custom operations and types (in its constructor).
/// It can also overridding general behavior of dialects exposed as virtual
/// method, for example regarding verification and parsing/printing.
class LiftDialect : public mlir::Dialect {
public:
    explicit LiftDialect(mlir::MLIRContext *ctx);

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

} // end namespace lift

#endif // MLIR_TUTORIAL_LIFT_DIALECT_H_
