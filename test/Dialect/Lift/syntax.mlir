module {
  func @test(%arg0: !lift.array) {
    %x = "bar"() : () -> !lift.array
    func @foo() -> !lift.float

    // Lift functionType:
//    %increment = !lift.function(%x, 5)



    "lift.return"() : () -> ()
  }
}