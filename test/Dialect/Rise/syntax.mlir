module {
  func @rise_id() {
    ^id:
        %42 = rise.literal 42 : !rise.int
        %id = rise.lambda %i : !rise.int -> !rise.int {
            rise.return %i
        }
        %result = rise.apply %id : !rise.fun<!rise.wrapped<int>, !rise.wrapped<int>>, %42

//        %xs = rise.array 5 !rise.nat
//        %arrayOfArrays = rise.array 200 !rise.array<2, array<1, array<5, float>>>


        "rise.return"() : () -> ()
//    "rise.return"(%id) : (!rise.fun<!rise.int, !rise.int>) -> ()
  }
//  func @dot_product(%m : !rise.array<5, !rise.nat>, %n : !rise.array<5, !rise.nat>) {
//      %zipped = rise.zip(%m, %n)
//      %xs = rise.apply(%addFun, %zipped)
//      %result = rise.reduce(%xs, %mulFun)
//      rise.return %result
//  }
}
