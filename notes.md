### Questions
* Should Lift Primitives be MLIR Ops?
#### TableGen
* execute here ..../llvm-project/llvm/projects/mlir/include/ with
`../../../cmake-build-debug/bin/mlir-tblgen --gen-op-decls mlir/Dialect/LoopOps/LoopOps.td`
otherwise includes are not found
* Different generators can be used (e.g `--gen-op-decls`, `--gen-op-defs`, `--gen-enum-defs`)
* This generates Ops for the LoopOps dialect. How is the then used in the dialect, should it be pasted into Ops.h? I can't find the generated output in the files for the LoopOps dialect.

### Types
* Types are registered in Types.h
* Complex types need a custom TypeStorage struct, which defines constraints for this type e.g. uniqueing -> these structs are defined in TypeDetail.h
* constants like `int x = 2` can be defined with the [_constant_](https://github.com/tensorflow/mlir/blob/master/g3doc/Dialects/Standard.md#constant-operation) operation e.g. `%x = lift.constant 42 : int`, or `%x = "lift.constant"()(value: 2} : int`

### Pitfalls when registering a new dialect:
* To introduce custim types it is neccessary to call `DEFINE_SYM_KIND_RANGE(MY_DIALECT)` in _include/mlir/IR/DialectSymbolRegistry.def_

### MLIR Questions
* regarding Toy-Ch3: passing an ill-structured .mlir file for the Toy dialect e.g  
    ```C++
    echo 'func @foo() -> !toy<"arra<1, 2, 3>">' | ./toyc-ch3 -emit=mlir -x=mlir
    ```
    to the Toy example correctly highlights the bad structure. However, an input which is meant for another language, even if incorrectly structured, is simply printed without errors. e.g 
    ```C++
    echo 'func @foo() -> !lift<"arra<1, 2, 3>">' | ./toyc-ch3 -emit=mlir -x=mlir
    ```
    Maybe fix this issue in the Toy example and create a Pullrequest?
