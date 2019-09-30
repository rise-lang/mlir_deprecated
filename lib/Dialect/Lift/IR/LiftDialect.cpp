//===- LiftDialect.cpp - Lift IR Dialect registration in MLIR ---------------===//
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
// This file implements the dialect for the Lift IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Lift/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlir {
namespace lift {

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
LiftDialect::LiftDialect(mlir::MLIRContext *ctx) : mlir::Dialect("lift", ctx) {
    addOperations<ConstantOp, GenericCallOp, PrintOp,
            TransposeOp, ReshapeOp,
            MulOp, AddOp, ReturnOp
//            ,
//#define GET_OP_LIST
//#include "mlir/Dialect/Lift/Ops.cpp.inc"
            >();
    addTypes<LiftArrayType, Kind, Nat, Data, Float, FunctionType>();
}





/// Parse a type registered to this dialect, we expect only Lift arrays.
mlir::Type LiftDialect::parseType(StringRef tyData, mlir::Location loc) const {
    // Sanity check: we only support array or array<...>

//  if (tyData.startswith("function")) {
//      return
//  }

    if (tyData.startswith("float")) {
//      tyData = tyData.drop_front(StringRef("float").size());
//
//      if (tyData.empty())
//          emitError(loc, "Float with no value given");
        //TODO: Check, that float is structured correctly
        return Float::get(getContext());
    }

    if (tyData.startswith("array")) {
        // Drop the "array" prefix from the type name, we expect either an empty
        // string or just the shape.
        tyData = tyData.drop_front(StringRef("array").size());
        // This is the generic array case without shape, early return it.
        if (tyData.empty())
            return LiftArrayType::get(getContext());

        // Use a regex to parse the shape (for efficient we should store this regex in
        // the dialect itself).
        SmallVector<StringRef, 4> matches;
        auto shapeRegex = llvm::Regex("^<([0-9]+)(, ([0-9]+))*>$");
        if (!shapeRegex.match(tyData, &matches)) {
            emitError(loc, "Invalid lift array shape '" + tyData + "'");
            return nullptr;
        }
        SmallVector<int64_t, 4> shape;
        // Iterate through the captures, skip the first one which is the full string.
        for (auto dimStr :
                llvm::make_range(std::next(matches.begin()), matches.end())) {
            if (dimStr.startswith(","))
                continue; // POSIX misses non-capturing groups.
            if (dimStr.empty())
                continue; // '*' makes it an optional group capture
            // Convert the capture to an integer
            unsigned long long dim;
            if (getAsUnsignedInteger(dimStr, /* Radix = */ 10, dim)) {
                emitError(loc, "Couldn't parse dimension as integer, matched: " + dimStr);
                return mlir::Type();
            }
            shape.push_back(dim);
        }
        // Finally we collected all the dimensions in the shape,
        // create the array type.
        return LiftArrayType::get(getContext(), shape);
    }


    emitError(loc, "Invalid Lift type '" + tyData + "', array expected");
    return nullptr;
}

/// Print a Lift array type, for example `array<2, 3, 4>`
void LiftDialect::printType(mlir::Type type, raw_ostream &os) const {

    switch (type.getKind()) {
        default: {
            os << "unknown lift type";
            return;
        }
        case LiftTypeKind::LIFT_ARRAY: {
            auto arrayTy = type.dyn_cast<LiftArrayType>();
            if (!arrayTy) {
                os << "unknown lift type";
                return;
            }
            os << "array";
            if (!arrayTy.getShape().empty()) {
                os << "<";
                mlir::interleaveComma(arrayTy.getShape(), os);
                os << ">";
            }
            break;
        }
        case LiftTypeKind::LIFT_FLOAT: {
            os << "float";
            break;
        }
    }
}

} //end namespace lift
} //end namespace mlir