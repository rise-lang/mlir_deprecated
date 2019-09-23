add_custom_target(Lift)
set_target_properties(Lift PROPERTIES FOLDER Examples)

macro(add_lift_target name)
  add_dependencies(Lift ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_lift_target name)



set(LLVM_LINK_COMPONENTS
        Support
        )

add_lift_target(lift-tut1
        liftTut1.cpp
        IR/AST.cpp
        IR/LiftDialect.cpp
        IR/MLIRGen.cpp
        IR/Ops.cpp
        IR/Types.cpp
        )
include_directories(include/)
target_link_libraries(lift-tut1
        PRIVATE
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRTransforms)
