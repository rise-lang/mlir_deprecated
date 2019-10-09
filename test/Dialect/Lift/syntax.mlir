module {
  func @lift_id() {
    ^id:
        %42 = lift.literal 42 : !lift.nat
        %id = lift.lambda %i : !lift.nat -> !lift.nat {
            lift.return %i
        }
        "lift.apply"(%42, %id) : (!lift.nat, !lift.lambda<!lift.nat, !lift.nat>) -> !lift.nat
    "lift.return"() : () -> ()
  }
}
