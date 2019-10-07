module {
  func @lift_program() {
    ^dot:
        %42 = "lift.literal"() {value = 42} : () -> !lift.nat
        %13 = "lift.literal"() {value = 13} : () -> !lift.nat
        //Lambda inputs have to be defined beforehand. We dont want this

        %add = "lift.lambda"()({
            //TODO: Add the capability to add nats
            "lift.return"() : () -> ()
        }){} : () -> !lift.lambda
        "lift.apply"(%42, %13, %add) : (!lift.nat, !lift.nat, !lift.lambda) -> !lift.nat

    "lift.return"() : () -> ()
  }
}