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
namespace rise {
namespace detail {


/// Rise type structure:
struct RiseDataTypeWrapperStorage : public mlir::TypeStorage {
    RiseDataTypeWrapperStorage(DataType data) : data(data) {}

    using KeyTy = DataType;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(data);
    }

    static KeyTy getKey(DataType data) {
        return KeyTy(data);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key.getAsOpaquePointer());
    }

    static RiseDataTypeWrapperStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
        return new(allocator.allocate<RiseFunTypeStorage>()) RiseDataTypeWrapperStorage(key);
    }

    DataType data;
};


struct RiseFunTypeStorage : public mlir::TypeStorage {
    RiseFunTypeStorage(RiseType input, RiseType output) : input(input), output(output) {}

    using KeyTy = std::pair<RiseType, RiseType>;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(input, output);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.first, key.second);
    }

    static KeyTy getKey(RiseType input, RiseType output) {
        return KeyTy(input, output);
    }

    static RiseFunTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
        return new(allocator.allocate<RiseFunTypeStorage>()) RiseFunTypeStorage(key.first, key.second);
    }

    RiseType input;
    RiseType output;
};

/// This class holds the implementation of the TupleType.
/// It is intended to be uniqued based on its content and owned by the context.
struct RiseTupleTypeStorage : public mlir::TypeStorage {
    RiseTupleTypeStorage(DataType first, DataType second) : first(first), second(second) {}
    /// This defines how we unique this type in the context: our key contains
    /// only the shape, a more complex type would have multiple entries in the
    /// tuple here.
    /// The element of the tuples usually matches 1-1 the arguments from the
    /// public `get()` method arguments from the facade.

    using KeyTy = std::pair<DataType, DataType>;

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.first, key.second);
    }

    /// When the key hash hits an existing type, we compare the shape themselves
    /// to confirm we have the right type.
    bool operator==(const KeyTy &key) const { return key == KeyTy(first, second); }


    static KeyTy getKey(DataType first, DataType second) {
        return KeyTy(first, second);
    }
    /// This is a factory method to create our type storage. It is only
    /// invoked after looking up the type in the context using the key and not
    /// finding it.
    static RiseTupleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
        return new(allocator.allocate<RiseTupleTypeStorage>()) RiseTupleTypeStorage(key.first, key.second);
    }

    DataType getFirst() const {return first; }
    DataType getSecond() const {return second; }
private:
    DataType first;
    DataType second;
};


/// This class holds the implementation of the RiseArrayType.
/// It is intended to be uniqued based on its content and owned by the context.
struct ArrayTypeStorage : public mlir::TypeStorage {
    ArrayTypeStorage(int size, DataType elementType) : size(size), elementType(elementType) {}
    /// This defines how we unique this type in the context: our key contains
    /// only the shape, a more complex type would have multiple entries in the
    /// tuple here.
    /// The element of the tuples usually matches 1-1 the arguments from the
    /// public `get()` method arguments from the facade.

    //TODO: array should only contain Rise types
    using KeyTy = std::pair<int, DataType>;

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.first, key.second);
    }

    /// When the key hash hits an existing type, we compare the shape themselves
    /// to confirm we have the right type.
    bool operator==(const KeyTy &key) const { return key == KeyTy(size, elementType); }


    static KeyTy getKey(int size, DataType elementType) {
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
    DataType getElementType() const { return elementType; }
private:
    int size;
    DataType elementType;
};

} //end namespace detail
} //end namespace rise
} //end namespace mlir
#endif //LLVM_TYPEDETAIL_H
