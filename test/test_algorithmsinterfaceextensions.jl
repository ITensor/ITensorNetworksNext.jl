import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @testset

# Define test problems, algorithms, and states for testing
struct TestProblem <: AIE.Problem
    data::Vector{Float64}
end

@kwdef struct TestAlgorithm{StoppingCriterion <: AI.StoppingCriterion} <: AIE.Algorithm
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(10)
end

@kwdef struct TestAlgorithmStep{StoppingCriterion <: AI.StoppingCriterion} <: AIE.Algorithm
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(5)
end

function AI.step!(
        problem::TestProblem, algorithm::TestAlgorithm, state::AIE.DefaultState;
        logging_context_prefix = Symbol()
    )
    state.iterate .+= 1  # Simple increment step
    return state
end

function AI.step!(
        problem::TestProblem, algorithm::TestAlgorithmStep, state::AIE.DefaultState;
        logging_context_prefix = Symbol()
    )
    state.iterate .+= 2  # Different increment step
    return state
end

@testset "AlgorithmsInterfaceExtensions" begin
    @testset "DefaultState" begin
        # Test DefaultState construction
        iterate = [1.0, 2.0, 3.0]
        stopping_criterion_state = AI.initialize_state(
            TestProblem([1.0]), TestAlgorithm(), TestAlgorithm().stopping_criterion
        )
        state = AIE.DefaultState(; iterate = copy(iterate), stopping_criterion_state)
        @test state.iterate == iterate
        @test state.iteration == 0
        @test state.stopping_criterion_state isa AI.StoppingCriterionState

        # Test DefaultState with custom iteration
        state.iteration = 5
        @test state.iteration == 5
    end

    @testset "initialize_state!" begin
        # Test initialize_state! with iterate kwarg
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm()
        stopping_criterion_state = AI.initialize_state(
            problem, algorithm, algorithm.stopping_criterion
        )
        state = AIE.DefaultState(; iterate = [0.0, 0.0], stopping_criterion_state)

        initial_iterate = [1.0, 2.0]
        AIE.AI.initialize_state!(problem, algorithm, state; iterate = initial_iterate)
        @test state.iterate == initial_iterate
        @test state.iteration == 0
    end

    @testset "initialize_state" begin
        # Test initialize_state without exclamation
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm()

        state = AIE.AI.initialize_state(problem, algorithm; iterate = [0.0, 0.0])
        @test state isa AIE.DefaultState
        @test state.iteration == 0
    end

    @testset "increment!" begin
        # Test increment! with problem and algorithm
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm()
        stopping_criterion_state = AI.initialize_state(
            problem, algorithm, algorithm.stopping_criterion
        )
        state = AIE.DefaultState(; iterate = [0.0, 0.0], stopping_criterion_state)

        # Increment and verify iteration counter increases
        AI.increment!(problem, algorithm, state)
        @test state.iteration == 1

        AI.increment!(problem, algorithm, state)
        @test state.iteration == 2
    end

    @testset "solve! and solve" begin
        # Test solve! with simple problem
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(3))

        initial_iterate = [10.0, 20.0]
        state = AI.initialize_state(problem, algorithm; iterate = copy(initial_iterate))

        # Solve with custom initial iterate
        final_state = AI.solve!(
            problem, algorithm, state; iterate = copy(initial_iterate)
        )

        @test final_state.iteration == 3
        # Each step increments by 1, so after 3 steps: [10, 20] + 3 = [13, 23]
        @test final_state.iterate ≈ [13.0, 23.0]

        # Test solve without exclamation
        problem2 = TestProblem([1.0, 2.0])
        algorithm2 = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(2))
        initial_iterate2 = [5.0, 10.0]

        final_state2 = AI.solve(problem2, algorithm2; iterate = copy(initial_iterate2))
        @test final_state2.iteration == 2
        @test final_state2.iterate ≈ [7.0, 12.0]
    end

    @testset "DefaultAlgorithmIterator" begin
        # Test algorithm iterator creation
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(2))
        initial_iterate = [0.0, 0.0]
        state = AI.initialize_state(problem, algorithm; iterate = copy(initial_iterate))
        iterator = AIE.algorithm_iterator(problem, algorithm, state)

        @test iterator isa AIE.DefaultAlgorithmIterator
        @test iterator.problem === problem
        @test iterator.algorithm === algorithm
        @test iterator.state === state

        # Test iteration interface
        @test !AI.is_finished!(iterator)

        # Step through iterator
        state_out, _ = iterate(iterator)
        @test state_out.iteration == 1
        @test state_out.iterate ≈ [1.0, 1.0]  # Incremented by step!

        state_out, _ = iterate(iterator)
        @test state_out.iteration == 2

        @test AI.is_finished!(iterator)
    end

    @testset "with_algorithmlogger" begin
        # Test with_algorithmlogger with functions
        results = []
        function callback1(problem, algorithm, state)
            push!(results, :callback1)
            return nothing
        end
        function callback2(problem, algorithm, state)
            push!(results, :callback2)
            return nothing
        end

        problem = TestProblem([1.0])
        algorithm = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(1))

        # Test with CallbackAction (wrapped functions)
        state = AIE.with_algorithmlogger(
            :TestProblem_TestAlgorithm_PreStep => callback1,
            :TestProblem_TestAlgorithm_PostStep => callback2,
        ) do
            return AI.solve(problem, algorithm; iterate = [0.0])
        end
        @test results == [:callback1, :callback2]
    end

    @testset "DefaultNestedAlgorithm" begin
        # Test creating nested algorithm with function
        nested_alg = AIE.nested_algorithm(3) do i
            return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
        end

        @test nested_alg isa AIE.DefaultNestedAlgorithm
        @test length(nested_alg.algorithms) == 3
        @test AIE.max_iterations(nested_alg) == 3

        # Test stepping through nested algorithm
        problem = TestProblem([1.0, 2.0])
        stopping_criterion_state = AI.initialize_state(
            problem, nested_alg, nested_alg.stopping_criterion
        )
        state = AIE.DefaultState(; iterate = [0.0, 0.0], stopping_criterion_state)

        initial_iterate = [0.0, 0.0]
        AI.solve!(
            problem, nested_alg, state; iterate = copy(initial_iterate)
        )

        @test state.iteration == 3
        # Each nested algorithm runs once with 2 steps, incrementing by 2
        # Total: 3 algorithms × 2 iterations × 2 increment = 12
        @test state.iterate ≈ [12.0, 12.0]
    end

    @testset "NestedAlgorithm basic tests" begin
        # Test basic nested algorithm functionality
        nested_alg = AIE.nested_algorithm(2) do i
            return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
        end

        problem = TestProblem([1.0, 2.0])

        # Test state initialization
        state_nested = AI.initialize_state(problem, nested_alg; iterate = [0.0, 0.0])

        @test state_nested isa AIE.DefaultState
        @test state_nested.iteration == 0
        @test AIE.max_iterations(nested_alg) == 2
    end

    @testset "increment! for nested algorithms" begin
        # Test increment! logic for nested algorithm state
        problem = TestProblem([1.0])
        nested_alg = AIE.nested_algorithm(2) do i
            return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
        end

        stopping_criterion_state = AI.initialize_state(
            problem, nested_alg, nested_alg.stopping_criterion
        )
        state = AIE.DefaultState(;
            iterate = [0.0],
            stopping_criterion_state = stopping_criterion_state,
        )

        # Test progression through iterations
        @test state.iteration == 0

        AI.increment!(problem, nested_alg, state)
        @test state.iteration == 1

        AI.increment!(problem, nested_alg, state)
        @test state.iteration == 2
    end

    @testset "get_subproblem and set_substate!" begin
        # Test get_subproblem
        problem = TestProblem([1.0, 2.0])
        nested_alg = AIE.nested_algorithm(2) do i
            return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(1))
        end

        stopping_criterion_state = AI.initialize_state(
            problem, nested_alg, nested_alg.stopping_criterion
        )
        state = AIE.DefaultState(;
            iterate = [5.0, 10.0],
            iteration = 1,
            stopping_criterion_state,
        )

        subproblem, subalgorithm, substate = AIE.get_subproblem(problem, nested_alg, state)
        @test subproblem === problem
        @test subalgorithm === nested_alg.algorithms[1]
        @test substate.iterate ≈ [5.0, 10.0]

        # Test set_substate!
        new_substate = AIE.DefaultState(;
            iterate = [100.0, 200.0],
            substate.stopping_criterion_state,
        )
        AIE.set_substate!(problem, nested_alg, state, new_substate)
        @test state.iterate ≈ [100.0, 200.0]
    end

    @testset "basetypenameof and default_logging_context_prefix" begin
        # Test basetypenameof utility
        problem = TestProblem([1.0])
        algorithm = TestAlgorithm()

        prefix_problem = AIE.default_logging_context_prefix(problem)
        prefix_algorithm = AIE.default_logging_context_prefix(algorithm)
        prefix_combined = AIE.default_logging_context_prefix(problem, algorithm)

        @test prefix_problem isa Symbol
        @test prefix_algorithm isa Symbol
        @test prefix_combined isa Symbol
        @test contains(String(prefix_combined), String(prefix_problem))
    end

    @testset "DefaultFlattenedAlgorithm" begin
        # Create nested algorithms that support max_iterations
        nested_algs = map(1:3) do i
            return AIE.nested_algorithm(1) do j
                return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
            end
        end

        flattened_alg = AIE.DefaultFlattenedAlgorithm(;
            algorithms = nested_algs,
            stopping_criterion = AI.StopAfterIteration(6) # 3 algorithms × 2 iterations each
        )

        @test flattened_alg isa AIE.DefaultFlattenedAlgorithm
        @test length(flattened_alg.algorithms) == 3

        # Test state initialization
        problem = TestProblem([1.0, 2.0])
        state_flat = AI.initialize_state(problem, flattened_alg; iterate = [0.0, 0.0])

        @test state_flat isa AIE.DefaultFlattenedAlgorithmState
        @test state_flat.iteration == 0
        @test state_flat.parent_iteration == 1
        @test state_flat.child_iteration == 0
    end

    @testset "DefaultFlattenedAlgorithmState increment!" begin
        # Create nested algorithms for flattened algorithm
        nested_algs = map(1:2) do i
            return AIE.nested_algorithm(1) do j
                return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
            end
        end

        flattened_alg = AIE.DefaultFlattenedAlgorithm(;
            algorithms = nested_algs,
            stopping_criterion = AI.StopAfterIteration(4),
        )

        problem = TestProblem([1.0])
        stopping_criterion_state = AI.initialize_state(
            problem, flattened_alg, flattened_alg.stopping_criterion
        )
        state = AIE.DefaultFlattenedAlgorithmState(;
            iterate = [0.0],
            stopping_criterion_state = stopping_criterion_state,
        )

        # Test initial state
        @test state.iteration == 0
        @test state.parent_iteration == 1
        @test state.child_iteration == 0

        # First increment - should increment child_iteration
        AI.increment!(problem, flattened_alg, state)
        @test state.iteration == 1
        @test state.parent_iteration == 1
        @test state.child_iteration == 1

        # Second increment - should increment child_iteration again
        AI.increment!(problem, flattened_alg, state)
        @test state.iteration == 2
        @test state.parent_iteration == 2  # Should move to next parent
        @test state.child_iteration == 1
    end

    @testset "FlattenedAlgorithm step!" begin
        # Test individual step! calls for flattened algorithm
        nested_algs = map(1:2) do i
            return AIE.nested_algorithm(1) do j
                return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
            end
        end

        flattened_alg = AIE.DefaultFlattenedAlgorithm(;
            algorithms = nested_algs,
            stopping_criterion = AI.StopAfterIteration(4)
        )

        problem = TestProblem([1.0, 2.0])
        state = AIE.AI.initialize_state(problem, flattened_alg; iterate = [0.0, 0.0])

        # Manually step through to test step! functionality
        AIE.AI.increment!(problem, flattened_alg, state)
        @test state.parent_iteration == 1
        @test state.child_iteration == 1

        AIE.AI.step!(problem, flattened_alg, state)
        # The nested algorithm runs TestAlgorithmStep with 2 iterations, each incrementing by 2
        @test state.iterate ≈ [4.0, 4.0]
    end

    @testset "flattened_algorithm helper" begin
        # Test the flattened_algorithm helper function
        nested_algs = map(1:2) do i
            return AIE.nested_algorithm(1) do j
                return TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
            end
        end

        # Using the helper function
        flattened_alg = AIE.flattened_algorithm(2) do i
            AIE.nested_algorithm(1) do j
                TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2))
            end
        end

        @test flattened_alg isa AIE.DefaultFlattenedAlgorithm
        @test length(flattened_alg.algorithms) == 2
    end

    @testset "AlgorithmIterator is_finished (without !)" begin
        # Test is_finished without mutation
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(1))
        initial_iterate = [0.0, 0.0]
        state = AI.initialize_state(problem, algorithm; iterate = copy(initial_iterate))
        iterator = AIE.algorithm_iterator(problem, algorithm, state)

        # Before any iterations
        @test !AI.is_finished(iterator)

        # Run the algorithm
        AI.solve!(problem, algorithm, state; iterate = copy(initial_iterate))

        # After completion
        @test AI.is_finished(iterator)
    end

    @testset "AlgorithmIterator step!" begin
        # Test step! method for iterator
        problem = TestProblem([1.0, 2.0])
        algorithm = TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(2))
        initial_iterate = [0.0, 0.0]
        state = AI.initialize_state(problem, algorithm; iterate = copy(initial_iterate))
        iterator = AIE.algorithm_iterator(problem, algorithm, state)

        # Step the iterator
        AI.step!(iterator)
        @test iterator.state.iterate ≈ [1.0, 1.0]

        AI.step!(iterator)
        @test iterator.state.iterate ≈ [2.0, 2.0]
    end

    @testset "NestedAlgorithm with different sub-algorithms" begin
        # Test nested algorithm with varying sub-algorithms
        nested_alg = AIE.DefaultNestedAlgorithm(;
            algorithms = [
                TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(1)),
                TestAlgorithmStep(; stopping_criterion = AI.StopAfterIteration(2)),
                TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(1)),
            ]
        )

        @test AIE.max_iterations(nested_alg) == 3
        @test length(nested_alg.algorithms) == 3

        problem = TestProblem([1.0, 2.0])
        state = AI.initialize_state(problem, nested_alg; iterate = [0.0, 0.0])

        AI.solve!(problem, nested_alg, state; iterate = [0.0, 0.0])

        # First algorithm: 1 iteration × 1 increment = 1
        # Second algorithm: 2 iterations × 2 increment = 4
        # Third algorithm: 1 iteration × 1 increment = 1
        # Total: 1 + 4 + 1 = 6
        @test state.iterate ≈ [6.0, 6.0]
        @test state.iteration == 3
    end

    @testset "Edge cases" begin
        # Test with single nested algorithm
        nested_alg = AIE.nested_algorithm(1) do i
            return TestAlgorithm(; stopping_criterion = AI.StopAfterIteration(1))
        end

        problem = TestProblem([1.0])
        state = AI.initialize_state(problem, nested_alg; iterate = [0.0])
        AI.solve!(problem, nested_alg, state; iterate = [0.0])

        @test state.iterate ≈ [1.0]
        @test state.iteration == 1
    end
end
