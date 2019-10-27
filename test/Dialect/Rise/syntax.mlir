module {
//    func @rise_id() {
//    ^id:
//        //integer literal
//        %42 = rise.literal #rise.lit<int<42>>
//        //Array demonstration
//        %array = rise.literal #rise.lit<array<2, !rise.int, [1,2]>>
//        %nestedArray = rise.literal #rise.lit<array<2.3, !rise.int, [[1,2,3],[4,5,6]]>>
//
//        %id = rise.lambda %i : !rise.int -> !rise.int {
//            rise.return %i : !rise.int
//        }
//        %result = rise.apply %id : !rise.fun<int -> int>, %42
//
//        "rise.return"() : () -> ()
////    "rise.return"(%id) : (!rise.fun<!rise.int, !rise.int>) -> ()
//    }
//    func @rise_add_example() {
//        %int0 = rise.literal #rise.lit<int<7>>
//        %int1 = rise.literal #rise.lit<int<13>>
//        // %int1 = rise.literal #rise.lit<!rise.int, 13> : !rise.dat<!rise.int>
//
//        %addFun = rise.lambda %summand0 : !rise.int -> !rise.fun<int -> int> {
//            %addWithSummand0 = rise.lambda %summand1 : !rise.int -> !rise.int {
//                %addition = rise.addi %summand0, %summand1
//                rise.return %addition : !rise.int
//            }
//            rise.return %addWithSummand0 : !rise.fun<int -> int>
//        }
//        %addWithInt0 = rise.apply %addFun : !rise.fun<int -> !rise.fun<int -> int>>, %int0
//        %result = rise.apply %addWithInt0 : !rise.fun<int -> int>, %int1
//
//        "rise.return"() : () -> ()
//    }
//    func @rise_tuple_example() {
//        //creating a simple tuple of an int and a float
//        %int0 = rise.literal #rise.lit<int<7>>
//        %float0 = rise.literal #rise.lit<float<13>>
//
//        %tupleFun = rise.tuple #rise.int #rise.float
//        %tupleWithInt0 = rise.apply %tupleFun : !rise.fun<data<int> -> fun<data<float> -> data<tuple<int, float>>>>, %int0
//        %tupleIntFloat = rise.apply %tupleWithInt0 : !rise.fun<data<float> -> data<tuple<int, float>>>, %float0
//
//        //zipping two arrays
//        %array0 = rise.literal #rise.lit<array<2, !rise.int, [1,2]>>
//        %array1 = rise.literal #rise.lit<array<2, !rise.int, [1,2]>>
//
//        %zipFun = rise.zip #rise.nat<2> #rise.int #rise.int //: !rise.Array<!rise.nat<2>, !rise.int> -> !rise.Array<!rise.nat<2>, !rise.int> -> !rise.Array<!rise.nat<2>, !rise.Tuple<!rise.int, !rise.int>>
//        %zipWithArray0 = rise.apply %zipFun : !rise.fun<data<array<2, int>> -> fun<data<array<2, int>> -> data<array<2, tuple<int, int>>>>>, %array0 //: !rise.Array<!rise.nat<2>, !rise.int> -> !rise.Array<!rise.nat<2>, !rise.Tuple<!rise.int, !rise.int>>
//        %zippedArrays  = rise.apply %zipWithArray0 : !rise.fun<data<array<2, int>> -> data<array<2, tuple<int, int>>>>, %array1 //: !rise.Array<!rise.nat<2>, !rise.Tuple<!rise.int, !rise.int>>
//
//
//
//        "rise.return"() : () -> ()
//    }
    func @rise_map_example() {
        %array = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
        %doubleFun = rise.lambda %summand : !rise.int -> !rise.int {
            %double = rise.addi %summand, %summand
            rise.return %double : !rise.int
        }
        %map10IntsToInts = rise.map #rise.nat<10> #rise.int #rise.int
        %mapDoubleFun = rise.apply %map10IntsToInts : !rise.fun<fun<data<int> -> data<int>> -> fun<data<array<10, int>> -> data<array<10, int>>>>, %doubleFun
        %doubledArray = rise.apply %mapDoubleFun : !rise.fun<data<array<10, int>> -> data<array<10, int>>>, %array

        "rise.return"() : () -> ()
    }



///proposed structure:
//        // Natural numbers: N =
//        !rise.nat<2>
//
//        // Data types: DT =
//        !rise.int
//        !rise.Array<N, DT>
//        !rise.Tuple<DT, DT>
//        !rise.natAsData<N>
//
//        // Rise Types: T =
//        !rise.fun<T, T>
//        !rise.data<DT>

//         // FUTURE with dependent functions (+ their types)
//         %z0 = rise.zip : (N: !rise.nat<2>) -> !rise.Array<N, > -> !rise.Array<N, > -> !rise.Array<N, >
//         %z1 = rise.depApply %z0 !rise.nat<2>
//         %z2 = rise.depApply %z1 !rise.int
//         %z3 = rise.depApply %z2 !rise.int
//         %z4 = rise.apply    %z3 %array0
//         %zz = rise.apply    %z4 %array1




//  func @rise_map_example() {
//          %array0 = rise.literal #rise.array<2, !rise.int, [1,2]>
//
//          %id = rise.lambda %i : !rise.int -> !rise.int {
//              rise.return %i : !rise.int
//          }
//
////          %arrayMapped = rise.map %id %array0
//
//           "rise.return"() : () -> ()
//    }
//  func @dot_product(%m : !rise.array<5, !rise.nat>, %n : !rise.array<5, !rise.nat>) {
//      %zipped = rise.zip(%m, %n)
//      %xs = rise.apply(%addFun, %zipped)
//      %result = rise.reduce(%xs, %mulFun)
//      rise.return %result
//  }
}
