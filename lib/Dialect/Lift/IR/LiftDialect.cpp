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
    addOperations<
            ConstantOp, GenericCallOp, PrintOp,
            TransposeOp, ReshapeOp,
            MulOp, AddOp,
#define GET_OP_LIST
#include "mlir/Dialect/Lift/Ops.cpp.inc"
    >();
    addTypes<Kind, Nat, Data, Float, LambdaType>();
}



//mlir::Type parseLambdaType(StringRef typeString, Location loc) {
//    if (!typeString.consume_front("lambda<") || !typeString.consume_back(">")) {
//        emitError(loc, "lift.lambda delimiter <...> mismatch");
//        return Type();
//    }
//    // Split into input type and output type
//    StringRef inputData, outputData;
//    std::tie(inputData, outputData) = typeString.rsplit(',');
//    if (outputData.empty()) {
//        emitError(loc,
//                  "expected comma to separate input type and output type '")
//                << typeString << "'";
//        return Type();
//    }
//    inputData = inputData.trim();
//    outputData = outputData.trim();
//
//    //Do further parsing to decide which types these are.
//    //for now always assume lift.nats
//
//    return LambdaType::get(getContext(), Nat::get(getContext()), Nat::get(getContext()));
//}


/// Parse a type registered to this dialect, we expect only Lift arrays.
mlir::Type LiftDialect::parseType(StringRef typeString, mlir::Location loc) const {


    if (typeString.startswith("lambda")) {
//        return parseLambdaType(typeString, loc);
        if (!typeString.consume_front("lambda<") || !typeString.consume_back(">")) {
            emitError(loc, "lift.lambda delimiter <...> mismatch");
            return Type();
        }
        // Split into input type and output type
        StringRef inputData, outputData;
        std::tie(inputData, outputData) = typeString.rsplit(',');
        if (outputData.empty()) {
            emitError(loc,
                      "expected comma to separate input type and output type '")
                    << typeString << "'";
            return Type();
        }
        inputData = inputData.trim();
        outputData = outputData.trim();

        //Do further parsing to decide which types these are.
        //for now always assume lift.nats

        return LambdaType::get(getContext(), Nat::get(getContext()), Nat::get(getContext()));
    }
//
//    if (tyData.startswith("fun")) {
////      return FunctionType::get(getContext());
//    }

    if (typeString.startswith("float")) {
//      tyData = tyData.drop_front(StringRef("float").size());
//
//      if (tyData.empty())
//          emitError(loc, "Float with no value given");
        //TODO: Check, that float is structured correctly
        return Float::get(getContext());
    }
    if (typeString.startswith("nat")) {
//      tyData = tyData.drop_front(StringRef("float").size());
//
//      if (tyData.empty())
//          emitError(loc, "Float with no value given");
        //TODO: Check, that float is structured correctly
        return Nat::get(getContext());
    }

    emitError(loc, "Invalid Lift type '" + typeString + "'");
    return nullptr;
}

/// Print a Lift type
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
        case LiftTypeKind::LIFT_NAT: {
            os << "nat";
            break;
        }
        case LiftTypeKind ::LIFT_LAMBDA: {
            os << "lambda<" << type.dyn_cast<LambdaType>().getInput()
                << ", " << type.dyn_cast<LambdaType>().getOutput()
                << ">";
            break;
        }
        case LiftTypeKind::LIFT_FUNCTIONTYPE: {
            os << "fun";
            break;
        }
    }
}

} //end namespace lift
} //end namespace mlir