import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @test_throws, @testset

# Concrete `NestedAlgorithm` subtype: holds a flat list of child algorithms
# and picks them by iteration index. Mirrors how `BeliefPropagationAlgorithm`
# shapes itself on top of the minimal `AIE.NestedAlgorithm`.
struct TestProblem <: AI.Problem end

@kwdef struct TestChildAlgorithm{StoppingCriterion <: AI.StoppingCriterion} <: AI.Algorithm
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(2)
end

@kwdef mutable struct TestChildState{SCState <: AI.StoppingCriterionState} <: AI.State
    iterate::Vector{Float64}
    iteration::Int = 0
    stopping_criterion_state::SCState
end

function AI.initialize_state(
        problem::TestProblem, algorithm::TestChildAlgorithm;
        iterate, kwargs...
    )
    sc_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return TestChildState(; iterate, stopping_criterion_state = sc_state, kwargs...)
end

function AI.initialize_state!(
        problem::TestProblem, algorithm::TestChildAlgorithm, state::TestChildState;
        iteration = 0, kwargs...
    )
    for (k, v) in pairs(kwargs)
        setproperty!(state, k, v)
    end
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AI.increment!(
        problem::TestProblem, algorithm::TestChildAlgorithm, state::TestChildState
    )
    return AI.increment!(state)
end

function AI.step!(
        problem::TestProblem, algorithm::TestChildAlgorithm, state::TestChildState
    )
    state.iterate .+= 1
    return state
end

@kwdef struct TestNestedAlgorithm{
        ChildAlgorithm <: AI.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end

# Reuse the child-state shape for the parent algorithm too.
function AI.initialize_state(
        problem::TestProblem, algorithm::TestNestedAlgorithm;
        iterate, kwargs...
    )
    sc_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return TestChildState(; iterate, stopping_criterion_state = sc_state, kwargs...)
end

function AI.initialize_state!(
        problem::TestProblem, algorithm::TestNestedAlgorithm, state::TestChildState;
        iteration = 0, kwargs...
    )
    for (k, v) in pairs(kwargs)
        setproperty!(state, k, v)
    end
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AI.increment!(
        problem::TestProblem, algorithm::TestNestedAlgorithm, state::TestChildState
    )
    return AI.increment!(state)
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
    @testset "NestedAlgorithm defaults" begin
        # The bare `initialize_subsolve` default throws a `MethodError`,
        # forcing concrete subtypes to provide their own override.
        problem = TestProblem()
        algorithm = TestChildAlgorithm()
        state = AI.initialize_state(problem, algorithm; iterate = [0.0])
        @test_throws MethodError AIE.initialize_subsolve(problem, algorithm, state)

        # `finalize_substate!` copies the substate's iterate back into the
        # parent state.
        substate = AI.initialize_state(problem, algorithm; iterate = [42.0])
        AIE.finalize_substate!(problem, algorithm, substate, state)
        @test state.iterate == [42.0]
    end

    @testset "TestNestedAlgorithm" begin
        problem = TestProblem()
        nested_alg = TestNestedAlgorithm(;
            algorithms = [
                TestChildAlgorithm(; stopping_criterion = AI.StopAfterIteration(1)),
                TestChildAlgorithm(; stopping_criterion = AI.StopAfterIteration(2)),
            ]
        )
        @test nested_alg isa AIE.NestedAlgorithm

        state = AI.initialize_state(problem, nested_alg; iterate = [0.0, 0.0])
        AI.solve!(problem, nested_alg, state; iterate = [0.0, 0.0])
        # Two child algorithms: 1 inner step + 2 inner steps = 3 total
        # `state.iterate .+= 1` calls.
        @test state.iteration == 2
        @test state.iterate ≈ [3.0, 3.0]
    end
end
