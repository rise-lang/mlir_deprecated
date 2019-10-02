

file(GLOB globbed *.c *.cpp)
add_llvm_library(LIFT
        ${globbed}

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Lift
        )
add_dependencies(LIFT MLIRLiftOpsIncGen LLVMSupport)
target_link_libraries(LIFT
    PRIVATE
        LLVMSupport
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRTransforms
        LLVMSupport)

#add_executable(lift_tut1 liftTut1.cpp
#        IR/AST.cpp
#        IR/LiftDialect.cpp
#        IR/MLIRGen.cpp
#        IR/Ops.cpp
#        IR/Types.cpp )
#include_directories(include/)
#add_dependencies(lift_tut1
#        MLIRLiftOpsIncGen
#        LLVMSupport)
#target_link_libraries(lift_tut1
#        PRIVATE
#        LIFT
#        LLVMSupport
#        MLIRAnalysis
#        MLIRIR
#        MLIRParser
#        MLIRTransforms
#        )
#
#add_custom_target(lift_tut1)
#set_target_properties(lift_tut1 PROPERTIES FOLDER Examples)
#add_dependencies(lift_tut1 LIFT)
#
macro(add_lift_target name)
  add_dependencies(LIFT ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_lift_target name)
#
#
#
#set(LLVM_LINK_COMPONENTS
#        Support
#        )
#

add_lift_target(lift-tut1
        liftTut1.cpp
        IR/AST.cpp
        IR/LiftDialect.cpp
        IR/MLIRGen.cpp
        IR/Ops.cpp
        IR/Types.cpp
        )
include_directories(include/)
#add_dependencies(Lift MLIRLiftOpsIncGen MLIRStandardOps LLVMSupport)
target_link_libraries(lift-tut1
        PRIVATE
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRTransforms
        LLVMSupport)
