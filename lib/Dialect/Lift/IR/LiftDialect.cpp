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

#include <iostream>
#include <sstream>
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
#define GET_OP_LIST
#include "mlir/Dialect/Lift/Ops.cpp.inc"
    >();
    addTypes<Kind, Nat, Data, Float, LambdaType, ArrayType>();
    addAttributes<LiftTypeAttr>();
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


/// Parse a type registered to this dialect
mlir::Type LiftDialect::parseType(StringRef typeString, mlir::Location loc) const {


    if (typeString.startswith("lambda")) {
//        return parseLambdaType(typeString, loc);
        if (!typeString.consume_front("lambda<") || !typeString.consume_back(">")) {
            emitError(loc, "lift.lambda delimiter <...> mismatch");
            return Type();
        }
        // Split into input type and output type
        StringRef inputDataString, outputDataString;
        std::tie(inputDataString, outputDataString) = typeString.rsplit(',');
        if (outputDataString.empty()) {
            emitError(loc,
                      "expected comma to separate input type and output type '")
                    << typeString << "'";
            return Type();
        }
        inputDataString = inputDataString.trim();
        outputDataString = outputDataString.trim();

        //We can specify `!lift.type` as well as just `type`
        if (inputDataString.startswith("!lift."))
            inputDataString.consume_front("!lift.");
        if (outputDataString.startswith("!lift."))
            outputDataString.consume_front("!lift.");

        Type inputData = parseType(inputDataString, loc);
        Type outputData = parseType(outputDataString, loc);

        return LambdaType::get(getContext(), inputData, outputData);
    }
    if (typeString.startswith("array") || typeString.startswith("!lift.array")) {
        std::cout << "full typeString: " << typeString.str() << "\n";
        if (!typeString.consume_front("array<") || !typeString.consume_back(">")) {
            emitError(loc, "lift.array delimiter <...> mismatch");
            return Type();
        }

        // Split into size and elementType at the first `,`
        StringRef sizeString, elementTypeString;
        std::tie(sizeString, elementTypeString) = typeString.split(',');
        if (elementTypeString.empty()) {
            emitError(loc,
                      "expected comma to separate size and elementType'")
                    << typeString << "'";
            return Type();
        }

        //getting rid of leading or trailing whitspaces etc.
        sizeString = sizeString.trim();
        elementTypeString = elementTypeString.trim();

        //this should work, because sizeString has already been parsed to be an int
        int size = std::stoi(sizeString);

        if (elementTypeString.startswith("!lift."))
            elementTypeString.consume_front("!lift.");
        Type elementType = parseType(elementTypeString, loc);

        std::cout << "have an array<" << size << ", " << elementTypeString.str() << "\n";

        return ArrayType::get(getContext(), size, elementType);
    }
    if (typeString.startswith("float")) {
        return Float::get(getContext());
    }
    if (typeString.startswith("nat")) {
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
        case LiftTypeKind::LIFT_FLOAT: {
            os << "float";
            break;
        }
        case LiftTypeKind::LIFT_NAT: {
            os << "nat";
            break;
        }
        case LiftTypeKind::LIFT_LAMBDA: {
            os << "lambda<" << type.dyn_cast<LambdaType>().getInput()
                << ", " << type.dyn_cast<LambdaType>().getOutput()
                << ">";
            break;
        }
        case LiftTypeKind::LIFT_ARRAY: {
            os << "array<" << type.dyn_cast<ArrayType>().getSize()
                << ", " << type.dyn_cast<ArrayType>().getElementType()
                << ">";
            break;
        }
        case LiftTypeKind::LIFT_FUNCTIONTYPE: {
            os << "fun";
            break;
        }
    }
}

mlir::Attribute LiftDialect::parseAttribute(llvm::StringRef attrString,
        mlir::Type type, mlir::Location loc) const {

    emitError(loc,
              "")
            << "I want to parse this attribute" << attrString;

}

void LiftDialect::printAttribute(Attribute attribute, raw_ostream &os) const {
    //Not sure this is the right way to do Arrays


    switch (attribute.getKind()) {
        default: {
            os << "unknown lift attribute";
            return;
        }
        case LiftAttributeKind::LIFT_TYPE_ATTR: {
            switch (attribute.dyn_cast<LiftTypeAttr>().getValue().getKind()) {
                default: {
                    os << "unknown lift type";
                    return;
                }
                case LiftTypeKind::LIFT_FLOAT: {
                    os << "float";
                    break;
                }
                case LiftTypeKind::LIFT_NAT: {
                    os << "nat";
                    break;
                }
                case LiftTypeKind::LIFT_LAMBDA: {
                    os << "lambda<" << attribute.dyn_cast<LiftTypeAttr>().getValue().dyn_cast<LambdaType>().getInput()
                       << ", " << attribute.dyn_cast<LiftTypeAttr>().getValue().dyn_cast<LambdaType>().getOutput()
                       << ">";
                    break;
                }
                case LiftTypeKind::LIFT_ARRAY: {
                    os << "array<" << attribute.dyn_cast<LiftTypeAttr>().getValue().dyn_cast<ArrayType>().getSize()
                       << ", " << attribute.dyn_cast<LiftTypeAttr>().getValue().dyn_cast<ArrayType>().getElementType()
                       << ">";
                    break;
                }
                case LiftTypeKind::LIFT_FUNCTIONTYPE: {
                    os << "fun";
                    break;
                }
            }
        }
    }
}

} //end namespace lift
} //end namespace mlir