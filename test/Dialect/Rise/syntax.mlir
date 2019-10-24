module {
  func @rise_id() {
    ^id:
        %42 = rise.literal #rise.int<42>
        //Array demonstration
        %array = rise.literal #rise.array<2, !rise.int, [1,2]>
        %nestedArray = rise.literal #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>

        %id = rise.lambda %i : !rise.int -> !rise.int {
            rise.return %i : !rise.int
        }
        %result = rise.apply %id : !rise.fun<int, int>, %42

        "rise.return"() : () -> ()
//    "rise.return"(%id) : (!rise.fun<!rise.int, !rise.int>) -> ()
  }
  func @rise_add_example() {
        %int0 = rise.literal #rise.int<7>
        %int1 = rise.literal #rise.int<13>

        %addFun = rise.lambda %summand0 : !rise.int -> !rise.fun<int, int> {
            %nested = rise.lambda %summand1 : !rise.int -> !rise.int {
                %addition = rise.addi %summand0, %summand1
                rise.return %addition : !rise.int
            }
            rise.return %nested : !rise.fun<int, int>
        }
        %add = rise.apply %addFun : !rise.fun<int, !rise.fun<int, int>>, %int0
        %result = rise.apply %add : !rise.fun<int, int>, %int1

        "rise.return"() : () -> ()
  }
  func @rise_zip_example() {
        %int0 = rise.literal #rise.int<7>
        %int1 = rise.literal #rise.float<13>
        %intTuple = rise.tuple %int0 : !rise.int, %int1 : !rise.float

        %array0 = rise.literal #rise.array<2, !rise.int, [1,2]>
        %array1 = rise.literal #rise.array<2, !rise.int, [1,2]>
        %arrayTuple = rise.zip %array0 : !rise.array<2, int>, %array1 : !rise.array<2, int>

         "rise.return"() : () -> ()
  }
//  func @dot_product(%m : !rise.array<5, !rise.nat>, %n : !rise.array<5, !rise.nat>) {
//      %zipped = rise.zip(%m, %n)
//      %xs = rise.apply(%addFun, %zipped)
//      %result = rise.reduce(%xs, %mulFun)
//      rise.return %result
//  }
}
