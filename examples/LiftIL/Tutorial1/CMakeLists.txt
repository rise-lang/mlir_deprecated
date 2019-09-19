set(LLVM_LINK_COMPONENTS
  Support
  )

add_lift_target(lift-tut1
        ../../../lib/Dialect/Lift/liftTut1.cpp
  parser/AST.cpp
  mlir/LiftDialect.cpp
  mlir/MLIRGen.cpp
  )
include_directories(include/)
target_link_libraries(lift-tut1
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRTransforms)
