### Pitfalls when registering a new dialect:
* To introduce custim types it is neccessary to call `DEFINE_SYM_KIND_RANGE(MY_DIALECT)` in _include/mlir/IR/DialectSymbolRegistry.def_

### Questions
* regarding Toy-Ch3: passing an ill-structured .mlir file for the Toy dialect e.g  
    ```C++
    echo 'func @foo() -> !toy<"arra<1, 2, 3>">' | ./toyc-ch3 -emit=mlir -x=mlir
    ```
    to the Toy example correctly highlights the bad structure. However, an input which is meant for another language, even if incorrectly structured, is simply printed without errors. e.g 
    ```C++
    echo 'func @foo() -> !lift<"arra<1, 2, 3>">' | ./toyc-ch3 -emit=mlir -x=mlir
    ```
    Maybe fix this issue in the Toy example and create a Pullrequest?
