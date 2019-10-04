module {
  func @test(%arg0: !lift.array) {
    %x = "bar"() : () -> !lift.array
    func @foo() -> !lift.float

//    "lift.print"(%x)
    // Lift functionType:
//    %increment = !lift.function(%x, 5)

//    %f = "lift.constant"(){value: 42} : i32
//    %test = "lift.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> !lift.array

    %y = "lift.literal"() {value = 42} : () -> !lift.nat
    %z = "lift.literal"() {value = 2} : () -> !lift.nat
    "lift.return"() : () -> ()
  }
}