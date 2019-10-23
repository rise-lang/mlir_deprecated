module {
  func @rise_id() {
    ^id:
        %42 = rise.literal #rise.int<42>
        %array = rise.literal #rise.array<2, !rise.int, [1,2]>
        %nestedArray = rise.literal #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>

        %id = rise.lambda %i : !rise.int -> !rise.int {
            rise.return %i
        }
        %result = rise.apply %id : !rise.fun<!rise.wrapped<int>, !rise.wrapped<int>>, %42

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
