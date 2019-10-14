module {
  func @test(!rise.nat) {


//    "rise.print"(%x)
    // Rise functionType:
//    %increment = !rise.function(%x, 5)

//    %f = "rise.constant"(){value: 42} : i32
//    %test = "rise.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> !rise.array


    ^literalTest(%arg0 : !rise.nat):
       %x = "bar"() : () -> !rise.array
       func @foo() -> !rise.float
       %y = "rise.literal"() {value = 42} : () -> !rise.nat
       %z = "rise.literal"() {value = 2} : () -> !rise.nat
       "rise.return"() : () -> ()

    ^lambdaTest:
        "rise.lambda"()({
            %test = "rise.literal"() {value = 42} : () -> !rise.nat
            "rise.terminator"() : () -> ()
        }) : () -> !rise.lambda
        "rise.return"() : () -> ()


    ^dot:
        %42 = "rise.literal"() {value = 42} : () -> !rise.nat
        %13 = "rise.literal"() {value = 13} : () -> !rise.nat
        //Lambda inputs have to be defined beforehand. We dont want this

        %add = "rise.lambda"()({
            //TODO: Add the capability to add nats
            "rise.terminator"() : () -> ()
        }){} : () -> !rise.lambda
        "rise.apply"(%42, %13, %add) : (!rise.nat, !rise.nat, !rise.lambda) -> !rise.nat

    "rise.return"() : () -> ()
  }
}