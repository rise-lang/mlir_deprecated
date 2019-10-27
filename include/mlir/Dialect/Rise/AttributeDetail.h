//
// Created by martin on 10/10/2019.
//

#ifndef LLVM_ATTRIBUTEDETAIL_H
#define LLVM_ATTRIBUTEDETAIL_H


#include "Attributes.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/IR/Types.h"
#include "Types.h"

namespace mlir {
namespace rise {
namespace detail {
/// An attribute representing a reference to a Rise type.

struct LiteralAttributeStorage : public mlir::AttributeStorage {
    LiteralAttributeStorage(DataType type, std::string value) : type(type), value(value) {}

    //This is intentionally a StringRef, hashing does not work with std::string for some reason
    using KeyTy = std::pair<DataType, llvm::StringRef>;

    /// Key equality function.
    bool operator==(const KeyTy &key) const { return key == KeyTy(type, value); }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(key.first, key.second);
    }

    /// Construct a new storage instance.
    static LiteralAttributeStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                               KeyTy key) {
        return new(allocator.allocate<LiteralAttributeStorage>())
                LiteralAttributeStorage(key.first, key.second);
    }

    DataType type;
    std::string value;
};


struct DataTypeAttributeStorage : public mlir::AttributeStorage {
    DataTypeAttributeStorage(DataType value) : value(value){}

    //This is intentionally a StringRef, hashing does not work with std::string for some reason
    using KeyTy = DataType;


    /// Key equality function.
    bool operator==(const KeyTy &key) const { return key == KeyTy(value); }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key.getAsOpaquePointer());
    }

    /// Construct a new storage instance.
    static DataTypeAttributeStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                               KeyTy key) {
        return new(allocator.allocate<DataTypeAttributeStorage>())
                DataTypeAttributeStorage(key);
    }

    DataType value;
};

struct RiseTypeAttributeStorage : public mlir::AttributeStorage {
    RiseTypeAttributeStorage(RiseType value) : value(value){}

    //This is intentionally a StringRef, hashing does not work with std::string for some reason
    using KeyTy = RiseType;


    /// Key equality function.
    bool operator==(const KeyTy &key) const { return key == KeyTy(value); }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key.getAsOpaquePointer());
    }
    /// Construct a new storage instance.
    static RiseTypeAttributeStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                               KeyTy key) {
        return new(allocator.allocate<RiseTypeAttributeStorage>())
                RiseTypeAttributeStorage(key);
    }

    RiseType value;
};

struct NatAttributeStorage : public mlir::AttributeStorage {
    NatAttributeStorage(int value) : value(value) {}

    using KeyTy = int;


    /// Key equality function.
    bool operator==(const KeyTy &key) const { return key == KeyTy(value); }


    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
    }

    /// Construct a new storage instance.
    static NatAttributeStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                               KeyTy key) {
        return new(allocator.allocate<NatAttributeStorage>()) NatAttributeStorage(key);
    }

    int value;
};
}
}
}

#endif //LLVM_ATTRIBUTEDETAIL_H
