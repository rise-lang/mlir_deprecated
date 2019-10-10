module {
  func @lift_id() {
    ^id:
        %42 = lift.literal 42 : !lift.nat
        %id = lift.lambda %i : !lift.nat -> !lift.nat {
            lift.return %i
        }
        "lift.apply"(%42, %id) : (!lift.nat, !lift.lambda<nat, nat>) -> !lift.nat

        %xs = lift.array 5 !lift.nat
        %arrayOfArrays = lift.array 200 !lift.array<2, array<1, array<5, float>>>



    "lift.return"() : () -> ()
  }
}
