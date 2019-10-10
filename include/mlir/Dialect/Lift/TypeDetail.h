//
// Created by martin on 23/09/2019.
//

#ifndef LLVM_TYPEDETAIL_H
#define LLVM_TYPEDETAIL_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
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
namespace detail {

struct LiftLambdaTypeStorage : public mlir::TypeStorage {
    LiftLambdaTypeStorage(Type input, Type output) : input(input), output(output) {}

    using KeyTy = std::pair<Type, Type>;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(input, output);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.first, key.second);
    }

    static KeyTy getKey(Type input, Type output) {
        return KeyTy(input, output);
    }

    static LiftLambdaTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
        return new(allocator.allocate<LiftLambdaTypeStorage>()) LiftLambdaTypeStorage(key.first, key.second);
    }

    Type input;
    Type output;
};


/// This class holds the implementation of the LiftArrayType.
/// It is intended to be uniqued based on its content and owned by the context.
struct ArrayTypeStorage : public mlir::TypeStorage {
    ArrayTypeStorage(int size, Type elementType) : size(size), elementType(elementType) {}
    /// This defines how we unique this type in the context: our key contains
    /// only the shape, a more complex type would have multiple entries in the
    /// tuple here.
    /// The element of the tuples usually matches 1-1 the arguments from the
    /// public `get()` method arguments from the facade.

    //TODO: array should only contain Lift types
    using KeyTy = std::pair<int, Type>;

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.first, key.second);
    }

    /// When the key hash hits an existing type, we compare the shape themselves
    /// to confirm we have the right type.
    bool operator==(const KeyTy &key) const { return key == KeyTy(size, elementType); }


    static KeyTy getKey(int size, Type elementType) {
        return KeyTy(size, elementType);
    }
    /// This is a factory method to create our type storage. It is only
    /// invoked after looking up the type in the context using the key and not
    /// finding it.
    static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
        return new(allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key.first, key.second);
    }

    int getSize() const { return size; }
    Type getElementType() const { return elementType; }
private:
    int size;
    Type elementType;
};

} //end namespace detail
} //end namespace lift
} //end namespace mlir
#endif //LLVM_TYPEDETAIL_H
