module {
    func @rise_id() {
    ^id:
        //integer literal
        %42 = rise.literal #rise.lit<int<42>>
        //Array demonstration
        %array = rise.literal #rise.lit<array<2, !rise.int, [1,2]>>
        %nestedArray = rise.literal #rise.lit<array<2.3, !rise.int, [[1,2,3],[4,5,6]]>>

        %id = rise.lambda (%i) : !rise.fun<data<int> -> data<int>> {
            rise.return %i : !rise.data<int>
        }
        %result = rise.apply %id : !rise.fun<int -> int>, %42


        %test = rise.lambda (%j, %k) : !rise.fun<data<int> -> fun<data<int> -> data<int>>> {
            rise.return %j : !rise.data<int>
        }

        "rise.return"() : () -> ()
//    "rise.return"(%id) : (!rise.fun<!rise.int, !rise.int>) -> ()
    }
//    func @rise_add_example() {
//        %int0 = rise.literal #rise.lit<int<7>>
//        %int1 = rise.literal #rise.lit<int<13>>
//        // %int1 = rise.literal #rise.lit<!rise.int, 13> : !rise.dat<!rise.int>
//
//        %addFun = rise.lambda (%summand0) : !rise.fun<data<int> -> fun<data<int> -> data<int>>> {
//            %addWithSummand0 = rise.lambda (%summand1) : !rise.fun<data<int> -> data<int>> {
////                %addition = rise.addi %summand0, %summand1
//                %addInternalFun = rise.add #rise.int
//                %addition = rise.apply %addInternalFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %summand0, %summand1
//                rise.return %addition : !rise.data<int>
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
//        %zipFun = rise.zip #rise.nat<2> #rise.int #rise.int
//        %zipWithArray0 = rise.apply %zipFun : !rise.fun<data<array<2, int>> -> fun<data<array<2, int>> -> data<array<2, tuple<int, int>>>>>, %array0
//        %zippedArrays  = rise.apply %zipWithArray0 : !rise.fun<data<array<2, int>> -> data<array<2, tuple<int, int>>>>, %array1
//
//        "rise.return"() : () -> ()
//    }
//
//    func @rise_map_example() {
//        %array = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
//        %doubleFun = rise.lambda (%summand) : !rise.fun<data<int> -> data<int>> {
////            %double = rise.addi %summand, %summandA
//            %addFun = rise.add #rise.int
//            %double = rise.apply %addFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %summand, %summand
//            rise.return %double : !rise.data<int>
//        }
//        %map10IntsToInts = rise.map #rise.nat<10> #rise.int #rise.int
//        %mapDoubleFun = rise.apply %map10IntsToInts : !rise.fun<fun<data<int> -> data<int>> -> fun<data<array<10, int>> -> data<array<10, int>>>>, %doubleFun
//        %doubledArray = rise.apply %mapDoubleFun : !rise.fun<data<array<10, int>> -> data<array<10, int>>>, %array
//
//        "rise.return"() : () -> ()
//    }
//
//    func @rise_reduce_example() {
//        %array = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
//        %addFun = rise.add #rise.int
//        %initializer = rise.literal #rise.lit<int<0>>
//
//        %reduce10Ints = rise.reduce #rise.nat<10> #rise.int #rise.int
//        %reduce10IntsAdd = rise.apply %reduce10Ints : !rise.fun<fun<data<int> -> fun<data<int> -> data<int>>> -> fun<data<int> -> fun<data<array<10, int>> -> data<int>>>>, %addFun
//        %reduce10IntsAddInitialized = rise.apply %reduce10IntsAdd : !rise.fun<data<int> -> fun<data<array<10, int>> -> data<int>>>, %initializer
//        %result = rise.apply %reduce10IntsAddInitialized : !rise.fun<data<array<10, int>> -> data<int>>, %array
//
//        "rise.return"() : () -> ()
//    }
//
//func @rise_dot_product() {
//    //Arrays
//    %array0 = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
//    %array1 = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
//
//    //Zipping
//    %zipFun = rise.zip #rise.nat<10> #rise.int #rise.int
//    %zippedArrays = rise.apply %zipFun : !rise.fun<data<array<10, int>> -> fun<data<array<10, int>> -> data<array<10, tuple<int, int>>>>>, %array0, %array1
//
//    //Multiply
//    %tupleMultFun = rise.lambda (%tuple) : !rise.fun<data<tuple<int, int>> -> data<int>> {
//        %fstFun = rise.fst #rise.int #rise.int
//        %sndFun = rise.snd #rise.int #rise.int
//
//        %fst = rise.apply %fstFun : !rise.fun<data<tuple<int, int>> -> data<int>> ,%tuple
//        %snd = rise.apply %sndFun : !rise.fun<data<tuple<int, int>> -> data<int>> ,%tuple
//
//        %multFun = rise.mult #rise.int
//        %result = rise.apply %multFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %snd, %fst
//
//        rise.return %result : !rise.data<int>
//    }
//    %map10TuplesToInts = rise.map #rise.nat<10> #rise.tuple<int, int> #rise.int
//    %multipliedArray = rise.apply %map10TuplesToInts : !rise.fun<fun<data<tuple<int, int>> -> data<int>> -> fun<data<array<10, tuple<int, int>>> -> data<array<10, int>>>>, %tupleMultFun, %zippedArrays
//
//    //Reduction
//    %addFun = rise.add #rise.int
//    %initializer = rise.literal #rise.lit<int<0>>
//    %reduce10Ints = rise.reduce #rise.nat<10> #rise.int #rise.int
//    %result = rise.apply %reduce10Ints : !rise.fun<fun<data<int> -> fun<data<int> -> data<int>>> -> fun<data<int> -> fun<data<array<10, int>> -> data<int>>>>, %addFun, %initializer, %multipliedArray
//
//    rise.return %result : !rise.data<int>
//}


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



}
