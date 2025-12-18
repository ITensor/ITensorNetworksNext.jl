import AlgorithmsInterface as AI
using ITensorNetworksNext: Region, Sweep, Sweeping
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @testset

struct TestProblem <: AIE.Problem
end

function AI.step!(problem::TestProblem, algorithm::Region, state::AIE.State; kwargs...)
    state.iterate = algorithm.region
    return state
end

@testset "Sweeping" begin
    @testset "Region" begin
        algorithm = Region("region"; foo = 1, bar = 2)
        @test algorithm isa AIE.NonIterativeAlgorithm
        @test algorithm isa AIE.Algorithm
        @test algorithm isa AI.Algorithm
        @test algorithm.region == "region"
        @test algorithm.kwargs == (; foo = 1, bar = 2)
        @test Region(; region = "region", foo = 1, bar = 2) == algorithm

        problem = TestProblem()
        iterate = ""
        state = AI.solve(problem, algorithm; iterate)
        @test state.iterate == "region"
    end
    @testset "Sweep" begin
    end
    @testset "Sweeping" begin
    end
end
