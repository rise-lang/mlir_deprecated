module {
  func @rise_id() {
    ^id:
        %42 = rise.literal #rise.int<42>
        %array = rise.literal #rise.array<2, !rise.int, [1,2]>
        %nestedArray = rise.literal #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>

        %id = rise.lambda %i : !rise.int -> !rise.int {
            rise.return %i :!rise.int
        }
        %result = rise.apply %id : !rise.fun<int, int>, %42

        "rise.return"() : () -> ()
//    "rise.return"(%id) : (!rise.fun<!rise.int, !rise.int>) -> ()
  }
  func @rise_add() {
        %summand0 = rise.literal #rise.int<7>
        %summand1 = rise.literal #rise.int<13>

        %add0 = rise.lambda %i : !rise.int -> !rise.fun<int, int> {
            %tmp = rise.lambda %j : !rise.int -> !rise.int {
                rise.return %i : !rise.int//+ %j
            }
            rise.return %tmp : !rise.fun<int, int>
        }
        %add = rise.apply %add0 : !rise.fun<int, !rise.fun<int, int>>, %summand0
        %result = rise.apply %add : !rise.fun<int, int>, %summand1

        "rise.return"() : () -> ()
  }
//  func @dot_product(%m : !rise.array<5, !rise.nat>, %n : !rise.array<5, !rise.nat>) {
//      %zipped = rise.zip(%m, %n)
//      %xs = rise.apply(%addFun, %zipped)
//      %result = rise.reduce(%xs, %mulFun)
//      rise.return %result
//  }
}
