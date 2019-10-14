//
// Created by martin on 10/10/2019.
//

#ifndef LLVM_ATTRIBUTEDETAIL_H
#define LLVM_ATTRIBUTEDETAIL_H


#include "Attributes.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace rise {
namespace detail {
/// An attribute representing a reference to a Rise type.
struct RiseTypeAttributeStorage : public mlir::AttributeStorage {
    using KeyTy = mlir::Type;

    RiseTypeAttributeStorage(mlir::Type value) : value(value) {}

    /// Key equality function.
    bool operator==(const KeyTy &key) const { return key == value; }

    /// Construct a new storage instance.
    static RiseTypeAttributeStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                               KeyTy key) {
        return new(allocator.allocate<RiseTypeAttributeStorage>())
                RiseTypeAttributeStorage(key);
    }

    mlir::Type value;
};

}
}
}

#endif //LLVM_ATTRIBUTEDETAIL_H
