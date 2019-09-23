//
// Created by martin on 2019-09-23.
//

#include "mlir/Dialect/Lift/Types.h"
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

namespace lift {
namespace detail {

/// This class holds the implementation of the LiftArrayType.
/// It is intended to be uniqued based on its content and owned by the context.
struct LiftArrayTypeStorage : public mlir::TypeStorage {
    /// This defines how we unique this type in the context: our key contains
    /// only the shape, a more complex type would have multiple entries in the
    /// tuple here.
    /// The element of the tuples usually matches 1-1 the arguments from the
    /// public `get()` method arguments from the facade.
    using KeyTy = std::tuple<ArrayRef<int64_t>>;

    static unsigned hashKey(const KeyTy &key) {
        return llvm::hash_combine(std::get<0>(key));
    }

    /// When the key hash hits an existing type, we compare the shape themselves
    /// to confirm we have the right type.
    bool operator==(const KeyTy &key) const { return key == KeyTy(getShape()); }

    /// This is a factory method to create our type storage. It is only
    /// invoked after looking up the type in the context using the key and not
    /// finding it.
    static LiftArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
        // Copy the shape array into the bumpptr allocator owned by the context.
        ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));

        // Allocate the instance for the LiftArrayTypeStorage itself
        auto *storage = allocator.allocate<LiftArrayTypeStorage>();
        // Initialize the instance using placement new.
        return new(storage) LiftArrayTypeStorage(shape);
    }

    ArrayRef<int64_t> getShape() const { return shape; }

private:
    ArrayRef<int64_t> shape;

    /// Constructor is only invoked from the `construct()` method above.
    LiftArrayTypeStorage(ArrayRef<int64_t> shape) : shape(shape) {}
};

} // namespace detail

mlir::Type LiftArrayType::getElementType() {
    return mlir::FloatType::getF64(getContext());
}

LiftArrayType LiftArrayType::get(mlir::MLIRContext *context,
                                 ArrayRef<int64_t> shape) {
    return Base::get(context, LiftTypeKind::LIFT_ARRAY, shape);
}

ArrayRef<int64_t> LiftArrayType::getShape() { return getImpl()->getShape(); }

} //namespace lift