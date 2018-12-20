@testset "allee extinction" begin
    global init =  setup([1.0 4.0 7.0;
                          2.0 5.0 8.0;
                          3.0 6.0 9.0])

    global output = ArrayOutput(init, 2)

    global model = Models(AlleeExtinction(minfounders = 8.0))

    sim!(output, model, init; tstop=2)
    @test output[1] == [ 1.0 4.0 7.0;
                         2.0 5.0 8.0;
                         3.0 6.0 9.0]

    @test output[2] == [ 0.0 0.0 0.0;
                         0.0 0.0 8.0;
                         0.0 0.0 9.0]
end
