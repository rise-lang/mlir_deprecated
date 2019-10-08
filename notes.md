#### Structuring of Types
* For constructing new Types there is a utility class `TypeBase` 
* This method prevents multiple inheritance which might be needed to model our Typesystem (not 100% sure about this)
* We could directly subclass the class `Type`


03. October:
### TableGen
#### the good
* now _finally_ working with the CMake build process
* generated files look similar to generated files from other dialects.
#### the bad (resolved)
* first we had undefined references to ArrayRef
* structuring of namespaces to mlir::lift around the import statements of generated files resolved this.
#### the ugly (to be resolved)
* now linking throws some undefined reference errors
* could be due to further namespace issues
* restructuring from mlir::lift::... to just mlir::... yields the same errors


old:
### Questions
* Should Lift Primitives be MLIR Ops? -probably yes
* In Standard_Dialect/Toy_Tutorial they use a `constant` operation to enable the definition of constants

* Do we have to extend the AST with Nodes to support Lift primitives? Or can they opssiby be modelled with `Expr_Call`?
Is this even what we want? I think so


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

### Pitfalls 
#### when registering a new dialect:
* To introduce custom types it is neccessary to call `DEFINE_SYM_KIND_RANGE(MY_DIALECT)` in _include/mlir/IR/DialectSymbolRegistry.def_
#### TableGen
* The include statement for a Table'Gen'd file e.g. `#include "mlir/Dialect/Lift/Ops.h.inc"` has to be inside `namespace  mlir`
#### custom operations
* operations with custom syntax must not be in "", otherwise the custom parser will never be called and `parseGenericOperation` will be used.

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
