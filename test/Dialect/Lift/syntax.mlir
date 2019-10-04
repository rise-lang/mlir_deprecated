module {
  func @test(!lift.nat) {


//    "lift.print"(%x)
    // Lift functionType:
//    %increment = !lift.function(%x, 5)

//    %f = "lift.constant"(){value: 42} : i32
//    %test = "lift.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> !lift.array


    ^literalTest(%arg0 : !lift.nat):
       %x = "bar"() : () -> !lift.array
       func @foo() -> !lift.float
       %y = "lift.literal"() {value = 42} : () -> !lift.nat
       %z = "lift.literal"() {value = 2} : () -> !lift.nat
       "lift.return"() : () -> ()

    ^lambdaTest:
        "lift.lambda"()({
            %test = "lift.literal"() {value = 42} : () -> !lift.nat
            "lift.terminator"() : () -> ()
        }) : () -> !lift.lambda
        "lift.return"() : () -> ()


    ^dot:
        %42 = "lift.literal"() {value = 42} : () -> !lift.nat
        %13 = "lift.literal"() {value = 13} : () -> !lift.nat
        //Lambda inputs have to be defined beforehand. We dont want this
        %add = "lift.lambda"()({
            //TODO: Add the capability to add nats
            "lift.terminator"() : () -> ()
        }) : () -> !lift.lambda
        "lift.apply"(%42, %13, %add) : (!lift.nat, !lift.nat, !lift.lambda) -> !lift.nat

    "lift.return"() : () -> ()
  }
}