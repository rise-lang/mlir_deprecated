file(GLOB globbed *.c *.cpp)
add_llvm_library(LIFT
        ${globbed}

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Lift
        )
add_dependencies(LIFT MLIRLiftOpsIncGen LLVMSupport)
target_link_libraries(LIFT LLVMSupport)




add_llvm_example(liftIL
        liftIL.cpp
        IR/AST.cpp
        IR/LiftDialect.cpp
        IR/MLIRGen.cpp
        IR/Ops.cpp
        IR/Attributes.cpp
        IR/Types.cpp)

include_directories(include/)
target_link_libraries(liftIL
        PRIVATE
        LIFT
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRTransforms
        LLVMSupport)
