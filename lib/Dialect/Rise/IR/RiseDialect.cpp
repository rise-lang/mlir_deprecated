//===- RiseDialect.cpp - Rise IR Dialect registration in MLIR ---------------===//
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
// This file implements the dialect for the Rise IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sstream>
#include "mlir/Dialect/Rise/Dialect.h"

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
namespace rise {

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
RiseDialect::RiseDialect(mlir::MLIRContext *ctx) : mlir::Dialect("rise", ctx) {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Rise/Ops.cpp.inc"
    >();
    //      Types:                              Nats:               Datatypes:
    addTypes<RiseType, FunType, DataTypeWrapper, Nat, NatTypeWrapper, DataType, Int, Float, ArrayType>();
    addAttributes<RiseTypeAttr>();
}



//mlir::Type parseLambdaType(StringRef typeString, Location loc) {
//    if (!typeString.consume_front("lambda<") || !typeString.consume_back(">")) {
//        emitError(loc, "rise.lambda delimiter <...> mismatch");
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
//    //for now always assume rise.nats
//
//    return LambdaType::get(getContext(), Nat::get(getContext()), Nat::get(getContext()));
//}
/// Parse a type registered to this dialect
mlir::Type RiseDialect::parseType(StringRef typeString, mlir::Location loc) const {

    if (typeString.startswith("fun") || typeString.startswith("wrapped")) {
        return parseRiseType(typeString, loc);
    }
    if (typeString.startswith("array") || typeString.startswith("int") || typeString.startswith("float")) {
        return parseDataType(typeString, loc);
    }
    if (typeString.startswith("nat")) {
        return parseNat(typeString, loc);
    }
    emitError(loc, "Invalid Rise type '" + typeString + "'");
    return nullptr;
}

RiseType RiseDialect::parseRiseType(StringRef typeString, mlir::Location loc) const {
    if (typeString.startswith("fun")) {
        if (!typeString.consume_front("fun<") || !typeString.consume_back(">")) {
            emitError(loc, "rise.fun delimiter <...> mismatch");
            return nullptr;
        }
        // Split into input type and output type
        StringRef inputDataString, outputDataString;
        std::tie(inputDataString, outputDataString) = typeString.rsplit(',');
        if (outputDataString.empty()) {
            emitError(loc,
                      "expected comma to separate input type and output type '")
                    << typeString << "'";
            return nullptr;
        }
        inputDataString = inputDataString.trim();
        outputDataString = outputDataString.trim();

        //We can specify `!rise.type` as well as just `type`
        if (inputDataString.startswith("!rise."))
            inputDataString.consume_front("!rise.");
        if (outputDataString.startswith("!rise."))
            outputDataString.consume_front("!rise.");

        RiseType inputData = parseRiseType(inputDataString, loc);
        RiseType outputData = parseRiseType(outputDataString, loc);

        return FunType::get(getContext(), inputData, outputData);
    }
    if (typeString.startswith("wrapped")) {
        if (!typeString.consume_front("wrapped<") || !typeString.consume_back(">")) {
            emitError(loc, "rise.wrapped delimiter <...> mismatch");
            return nullptr;
        }
        DataType wrappedType = parseDataType(typeString, loc);

        return DataTypeWrapper::get(getContext(), wrappedType);

    }
}

DataType RiseDialect::parseDataType(StringRef typeString, mlir::Location loc) const {
    if (typeString.startswith("array") || typeString.startswith("!rise.array")) {
//        std::cout << "full typeString: " << typeString.str() << "\n";
        if (!typeString.consume_front("array<") || !typeString.consume_back(">")) {
            emitError(loc, "rise.array delimiter <...> mismatch");
            return nullptr;
        }

        // Split into size and elementType at the first `,`
        StringRef sizeString, elementTypeString;
        std::tie(sizeString, elementTypeString) = typeString.split(',');
        if (elementTypeString.empty()) {
            emitError(loc,
                      "expected comma to separate size and elementType'")
                    << typeString << "'";
            return nullptr;
        }

        //getting rid of leading or trailing whitspaces etc.
        sizeString = sizeString.trim();
        elementTypeString = elementTypeString.trim();

        //this should work, because sizeString has already been parsed to be an int
        int size = std::stoi(sizeString);

        if (elementTypeString.startswith("!rise."))
            elementTypeString.consume_front("!rise.");
        mlir::Type elementType = parseType(elementTypeString, loc);

//        std::cout << "have an array<" << size << ", " << elementTypeString.str() << "\n";

        return ArrayType::get(getContext(), size, elementType);
    }
    if (typeString.startswith("float")) {
        return Float::get(getContext());
    }

    if (typeString.startswith("int")) {
        return Int::get(getContext());
    }
}

Nat RiseDialect::parseNat(StringRef typeString, mlir::Location loc) const {
    if (typeString.startswith("nat")) {
        return Nat::get(getContext());
    }
}




std::string static stringForType(Type type) {
    switch (type.getKind()) {
        default: {
            return "unknown rise type";
        }
        case RiseTypeKind::RISE_FLOAT: {
            return "float";
        }
        case RiseTypeKind::RISE_INT: {
            return "int";
        }
        case RiseTypeKind::RISE_NAT: {
            return "nat";
        }
        case RiseTypeKind::RISE_ARRAY: {
            return "array<" + std::to_string(type.dyn_cast<ArrayType>().getSize()) + ", "
            + stringForType(type.dyn_cast<ArrayType>().getElementType()) + ">";
        }
        case RiseTypeKind::RISE_FUNTYPE: {
            return "fun<"  + stringForType(type.dyn_cast<FunType>().getInput()) + ", "
            + stringForType(type.dyn_cast<FunType>().getOutput()) + ">";
        }
       case RiseTypeKind::RISE_DATATYPE_WRAPPER: {
            return "wrapped<" + stringForType(type.dyn_cast<DataTypeWrapper>().getData()) + ">";
        }
    }
}

/// Print a Rise type
void RiseDialect::printType(mlir::Type type, raw_ostream &os) const {
    os << stringForType(type);
}



mlir::Attribute RiseDialect::parseAttribute(llvm::StringRef attrString,
        mlir::Type type, mlir::Location loc) const {

    emitError(loc,
              "")
            << "I want to parse this attribute" << attrString;

}

void RiseDialect::printAttribute(Attribute attribute, raw_ostream &os) const {
    //Not sure this is the right way to do Arrays


    switch (attribute.getKind()) {
        default: {
            os << "unknown rise attribute";
            return;
        }
        case RiseAttributeKind::RISE_TYPE_ATTR: {
            switch (attribute.dyn_cast<RiseTypeAttr>().getValue().getKind()) {
                default: {
                    os << "unknown rise type";
                    return;
                }
                case RiseTypeKind::RISE_FLOAT: {
                    os << "float";
                    break;
                }
                case RiseTypeKind::RISE_NAT: {
                    os << "nat";
                    break;
                }
                case RiseTypeKind::RISE_LAMBDA: {
                    os << "lambda<" << attribute.dyn_cast<RiseTypeAttr>().getValue().dyn_cast<LambdaType>().getInput()
                       << ", " << attribute.dyn_cast<RiseTypeAttr>().getValue().dyn_cast<LambdaType>().getOutput()
                       << ">";
                    break;
                }
                case RiseTypeKind::RISE_ARRAY: {
                    os << "array<" << attribute.dyn_cast<RiseTypeAttr>().getValue().dyn_cast<ArrayType>().getSize()
                       << ", " << attribute.dyn_cast<RiseTypeAttr>().getValue().dyn_cast<ArrayType>().getElementType()
                       << ">";
                    break;
                }
                case RiseTypeKind::RISE_FUNTYPE: {
                    os << "fun";
                    break;
                }
            }
        }
    }
}

} //end namespace rise
} //end namespace mlir