file(GLOB globbed *.c *.cpp)
add_llvm_library(RISE
        ${globbed}

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise
        )
add_dependencies(RISE MLIRRiseOpsIncGen LLVMSupport)
target_link_libraries(RISE LLVMSupport)




add_llvm_example(riseIL
        riseIL.cpp
        IR/AST.cpp
        IR/RiseDialect.cpp
        IR/MLIRGen.cpp
        IR/Ops.cpp
        IR/Attributes.cpp
        IR/Types.cpp)

include_directories(include/)
target_link_libraries(riseIL
        PRIVATE
        RISE
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRTransforms
        LLVMSupport)
