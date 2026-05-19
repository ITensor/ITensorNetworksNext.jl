import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @test_throws, @testset

# Define test problems, algorithms, and states for testing
struct TestProblem <: AIE.Problem
    data::Vector{Float64}
end

@kwdef struct TestAlgorithm{StoppingCriterion <: AI.StoppingCriterion} <: AIE.Algorithm
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(10)
end

function AI.step!(
        problem::TestProblem, algorithm::TestAlgorithm, state::AIE.DefaultState
    )
    state.iterate .+= 1  # Simple increment step
    return state
end

# Concrete `NestedAlgorithm` subtype: holds a flat list of child algorithms
# and picks them by iteration index. Mirrors how `BeliefPropagation` shapes
# itself on top of the new minimal `AIE.NestedAlgorithm`.
@kwdef struct TestNestedAlgorithm{
        ChildAlgorithm <: AIE.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end

function AIE.initialize_subsolve(
        problem::TestProblem, algorithm::TestNestedAlgorithm, state::AI.State
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]
    substate = AI.initialize_state(subproblem, subalgorithm; state.iterate)
    return subproblem, subalgorithm, substate
end

@testset "AlgorithmsInterfaceExtensions" begin
    @testset "DefaultState" begin
        iterate = [1.0, 2.0, 3.0]
        stopping_criterion_state = AI.initialize_state(
            TestProblem([1.0]), TestAlgorithm(), TestAlgorithm().stopping_criterion
        )
        state = AIE.DefaultState(; iterate = copy(iterate), stopping_criterion_state)
        @test state.iterate == iterate
        @test state.iteration == 0
        @test state.stopping_criterion_state isa AI.StoppingCriterionState

        state.iteration = 5
        @test state.iteration == 5
    end

    @testset "initialize_state!" begin
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm()
        stopping_criterion_state = AI.initialize_state(
            problem, algorithm, algorithm.stopping_criterion
        )
        state = AIE.DefaultState(;
            iteration = 2, iterate = [0.0, 0.0], stopping_criterion_state
        )
        AI.initialize_state!(problem, algorithm, state)
        @test state.iterate == [0.0, 0.0]
        @test state.iteration == 0
        @test state.stopping_criterion_state == stopping_criterion_state
    end

    @testset "initialize_state" begin
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm()
        state = AI.initialize_state(problem, algorithm; iterate = [0.0, 0.0])
        @test state isa AIE.DefaultState
        @test state.iteration == 0
    end

    @testset "increment!" begin
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm()
        stopping_criterion_state = AI.initialize_state(
            problem, algorithm, algorithm.stopping_criterion
        )
        state = AIE.DefaultState(; iterate = [0.0, 0.0], stopping_criterion_state)

        AI.increment!(problem, algorithm, state)
        @test state.iteration == 1
        AI.increment!(problem, algorithm, state)
        @test state.iteration == 2
    end

    @testset "solve! and solve" begin
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(3))
        state = AI.initialize_state(problem, algorithm; iterate = [10.0, 20.0])

        initial_iterate = [5.0, 10.0]
        final_iterate = AI.solve!(
            problem, algorithm, state; iterate = copy(initial_iterate)
        )
        @test state.iteration == 3
        @test final_iterate == state.iterate
        # Each step increments by 1, so after 3 steps: [5, 10] + 3 = [8, 13]
        @test state.iterate ≈ [8.0, 13.0]

        problem2 = TestProblem([1.0, 2.0])
        algorithm2 = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(2))
        final_iterate2 = AI.solve(problem2, algorithm2; iterate = [5.0, 10.0])
        @test final_iterate2 ≈ [7.0, 12.0]
    end

    @testset "NestedAlgorithm defaults" begin
        # The bare `initialize_subsolve` default throws a `MethodError`,
        # forcing concrete subtypes to provide their own override.
        problem = TestProblem([1.0])
        algorithm = TestAlgorithm()
        state = AIE.DefaultState(;
            iterate = [0.0],
            stopping_criterion_state = AI.initialize_state(
                problem, algorithm, algorithm.stopping_criterion
            )
        )
        @test_throws MethodError AIE.initialize_subsolve(problem, algorithm, state)

        # `finalize_substate!` copies the substate's iterate back into the
        # parent state.
        substate = AIE.DefaultState(;
            iterate = [42.0],
            stopping_criterion_state = state.stopping_criterion_state
        )
        AIE.finalize_substate!(problem, algorithm, state, substate)
        @test state.iterate == [42.0]
    end

    @testset "TestNestedAlgorithm" begin
        problem = TestProblem([1.0, 2.0])
        nested_alg = TestNestedAlgorithm(;
            algorithms = [
                TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(1)),
                TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(2)),
            ]
        )
        @test nested_alg isa AIE.NestedAlgorithm

        state = AI.initialize_state(problem, nested_alg; iterate = [0.0, 0.0])
        AI.solve!(problem, nested_alg, state; iterate = [0.0, 0.0])
        # Two child algorithms: 1 iter + 2 iter = 3 inner increments total.
        @test state.iteration == 2
        @test state.iterate ≈ [3.0, 3.0]
    end
end
