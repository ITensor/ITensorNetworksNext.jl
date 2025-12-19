import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @testset

struct TestProblem <: AIE.Problem
end

struct TestRegion{R, Kwargs <: NamedTuple} <: AIE.NonIterativeAlgorithm
    region::R
    kwargs::Kwargs
end
TestRegion(region; kwargs...) = TestRegion(region, (; kwargs...))

function AI.solve!(problem::TestProblem, algorithm::TestRegion, state::AIE.State; kwargs...)
    new_iterate = (; algorithm.region, algorithm.kwargs.foo, algorithm.kwargs.bar)
    state.iterate = [state.iterate; [new_iterate]]
    return state
end

@testset "Sweeping" begin
    @testset "TestRegion" begin
        algorithm = TestRegion("region"; foo = 1, bar = 2)
        @test algorithm isa AIE.NonIterativeAlgorithm
        @test algorithm isa AIE.Algorithm
        @test algorithm isa AI.Algorithm
        @test algorithm.region == "region"
        @test algorithm.kwargs == (; foo = 1, bar = 2)

        problem = TestProblem()
        iterate = []
        state = AI.solve(problem, algorithm; iterate)
        @test state.iterate == [(; region = "region", foo = 1, bar = 2)]
    end
    @testset "Sweep" begin
        algorithm = AIE.nested_algorithm(3) do i
            return TestRegion("region$i"; foo = i, bar = 2i)
        end
        problem = TestProblem()
        iterate = []
        state = AI.solve(problem, algorithm; iterate)
        @test state.iterate == [
            (; region = "region1", foo = 1, bar = 2),
            (; region = "region2", foo = 2, bar = 4),
            (; region = "region3", foo = 3, bar = 6),
        ]
    end
    @testset "Sweeping" begin
        algorithm = AIE.nested_algorithm(2) do i
            AIE.nested_algorithm(3) do j
                return TestRegion("sweep$i, region$j"; foo = (i, j), bar = (2i, 2j))
            end
        end
        problem = TestProblem()
        iterate = []
        state = AI.solve(problem, algorithm; iterate)
        @test state.iterate == [
            (; region = "sweep1, region1", foo = (1, 1), bar = (2, 2)),
            (; region = "sweep1, region2", foo = (1, 2), bar = (2, 4)),
            (; region = "sweep1, region3", foo = (1, 3), bar = (2, 6)),
            (; region = "sweep2, region1", foo = (2, 1), bar = (4, 2)),
            (; region = "sweep2, region2", foo = (2, 2), bar = (4, 4)),
            (; region = "sweep2, region3", foo = (2, 3), bar = (4, 6)),
        ]
    end
end
