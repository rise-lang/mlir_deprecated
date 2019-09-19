add_custom_target(Lift)
set_target_properties(Lift PROPERTIES FOLDER Examples)

macro(add_lift_target name)
  add_dependencies(Lift ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_lift_target name)

add_subdirectory(Tutorial1)

