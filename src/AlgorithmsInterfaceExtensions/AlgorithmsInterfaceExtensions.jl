module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI #========================== Patches for AlgorithmsInterface.jl ============================#

abstract type Problem <: AI.Problem end
abstract type Algorithm <: AI.Algorithm end
abstract type State <: AI.State end

function AI.initialize_state!(
        problem::Problem, algorithm::Algorithm, state::State; iteration = 0, kwargs...
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

function AI.initialize_state(
        problem::Problem, algorithm::Algorithm; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return DefaultState(; stopping_criterion_state, kwargs...)
end #============================ DefaultState ================================================#

@kwdef mutable struct DefaultState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end #============================ increment! ==================================================#

# Custom version of `increment!` that also takes the problem and algorithm as arguments.
function AI.increment!(problem::Problem, algorithm::Algorithm, state::State)
    return AI.increment!(state)
end #============================ solve! ======================================================#

# Custom version of `solve!` that allows specifying the logger and also overloads
# `increment!` on the problem and algorithm.
function basetypenameof(x)
    return Symbol(last(split(String(Symbol(Base.typename(typeof(x)).wrapper)), ".")))
end
default_logging_context_prefix(x) = Symbol(basetypenameof(x), :_)
function default_logging_context_prefix(problem::Problem, algorithm::Algorithm)
    return Symbol(
        default_logging_context_prefix(problem),
        default_logging_context_prefix(algorithm)
    )
end
function AI.solve!(
        problem::Problem, algorithm::Algorithm, state::State;
        logging_context_prefix = default_logging_context_prefix(problem, algorithm),
        kwargs...
    )
    logger = AI.algorithm_logger()

    context_suffixes = [:Start, :PreStep, :PostStep, :Stop]
    contexts = Dict(context_suffixes .=> Symbol.(logging_context_prefix, context_suffixes))

    # initialize the state and emit message
    AI.initialize_state!(problem, algorithm, state; kwargs...)
    AI.emit_message(logger, problem, algorithm, state, contexts[:Start])

    # main body of the algorithm
    while !AI.is_finished!(problem, algorithm, state)
        AI.increment!(problem, algorithm, state)

        # logging event between convergence check and algorithm step
        AI.emit_message(logger, problem, algorithm, state, contexts[:PreStep])

        # algorithm step
        AI.step!(problem, algorithm, state; logging_context_prefix)

        # logging event between algorithm step and convergence check
        AI.emit_message(logger, problem, algorithm, state, contexts[:PostStep])
    end

    # emit message about finished state
    AI.emit_message(logger, problem, algorithm, state, contexts[:Stop])
    return state
end

function AI.solve(
        problem::Problem, algorithm::Algorithm;
        logging_context_prefix = default_logging_context_prefix(problem, algorithm),
        kwargs...
    )
    state = AI.initialize_state(problem, algorithm; kwargs...)
    return AI.solve!(problem, algorithm, state; logging_context_prefix, kwargs...)
end #============================ AlgorithmIterator ===========================================#

abstract type AlgorithmIterator end

function algorithm_iterator(
        problem::Problem, algorithm::Algorithm, state::State
    )
    return DefaultAlgorithmIterator(problem, algorithm, state)
end

function AI.is_finished!(iterator::AlgorithmIterator)
    return AI.is_finished!(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.is_finished(iterator::AlgorithmIterator)
    return AI.is_finished(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.increment!(iterator::AlgorithmIterator)
    return AI.increment!(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.step!(iterator::AlgorithmIterator)
    return AI.step!(iterator.problem, iterator.algorithm, iterator.state)
end
function Base.iterate(iterator::AlgorithmIterator, init = nothing)
    AI.is_finished!(iterator) && return nothing
    AI.increment!(iterator)
    AI.step!(iterator)
    return iterator.state, nothing
end

struct DefaultAlgorithmIterator{Problem, Algorithm, State} <: AlgorithmIterator
    problem::Problem
    algorithm::Algorithm
    state::State
end #============================ with_algorithmlogger ========================================#

# Allow passing functions, not just CallbackActions.
@inline function with_algorithmlogger(f, args::Pair{Symbol, AI.LoggingAction}...)
    return AI.with_algorithmlogger(f, args...)
end
@inline function with_algorithmlogger(f, args::Pair{Symbol}...)
    return AI.with_algorithmlogger(f, (first.(args) .=> AI.CallbackAction.(last.(args)))...)
end #============================ NestedAlgorithm =============================================#

abstract type NestedAlgorithm <: Algorithm end

function nested_algorithm(f::Function, nalgorithms::Int; kwargs...)
    return DefaultNestedAlgorithm(f, nalgorithms; kwargs...)
end

max_iterations(algorithm::NestedAlgorithm) = length(algorithm.algorithms)

function get_subproblem(
        problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]
    substate = AI.initialize_state(subproblem, subalgorithm; state.iterate)
    return subproblem, subalgorithm, substate
end

function set_substate!(
        problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State, substate::AI.State
    )
    state.iterate = substate.iterate
    return state
end

function AI.step!(
        problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State;
        logging_context_prefix = Symbol()
    )
    # Get the subproblem, subalgorithm, and substate.
    subproblem, subalgorithm, substate = get_subproblem(problem, algorithm, state)

    # Solve the subproblem with the subalgorithm.
    logging_context_prefix = Symbol(
        logging_context_prefix, default_logging_context_prefix(subalgorithm)
    )
    AI.solve!(subproblem, subalgorithm, substate; logging_context_prefix)

    # Update the state with the substate.
    set_substate!(problem, algorithm, state, substate)

    return state
end

#=
    DefaultNestedAlgorithm(sweeps::AbstractVector{<:Algorithm})

An algorithm that consists of running an algorithm at each iteration
from a list of stored algorithms.
=#
@kwdef struct DefaultNestedAlgorithm{
        ChildAlgorithm <: Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end
function DefaultNestedAlgorithm(f::Function, nalgorithms::Int; kwargs...)
    return DefaultNestedAlgorithm(; algorithms = f.(1:nalgorithms), kwargs...)
end #============================ FlattenedAlgorithm ==========================================#

# Flatten a nested algorithm.
abstract type FlattenedAlgorithm <: Algorithm end
abstract type FlattenedAlgorithmState <: State end

function flattened_algorithm(f::Function, nalgorithms::Int; kwargs...)
    return DefaultFlattenedAlgorithm(f, nalgorithms; kwargs...)
end

function AI.initialize_state(
        problem::Problem, algorithm::FlattenedAlgorithm; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return DefaultFlattenedAlgorithmState(; stopping_criterion_state, kwargs...)
end
function AI.increment!(
        problem::Problem, algorithm::Algorithm, state::FlattenedAlgorithmState
    )
    # Increment the total iteration count.
    state.iteration += 1
    # TODO: Use `is_finished!` instead?
    if state.child_iteration ≥ max_iterations(algorithm.algorithms[state.parent_iteration])
        # We're on the last iteration of the child algorithm, so move to the next
        # child algorithm.
        state.parent_iteration += 1
        state.child_iteration = 1
    else
        # Iterate the child algorithm.
        state.child_iteration += 1
    end
    return state
end
function AI.step!(
        problem::AI.Problem, algorithm::FlattenedAlgorithm, state::FlattenedAlgorithmState;
        logging_context_prefix = Symbol()
    )
    algorithm_sweep = algorithm.algorithms[state.parent_iteration]
    state_sweep = AI.initialize_state(
        problem, algorithm_sweep;
        state.iterate, iteration = state.child_iteration
    )
    AI.step!(problem, algorithm_sweep, state_sweep; logging_context_prefix)
    state.iterate = state_sweep.iterate
    return state
end

@kwdef struct DefaultFlattenedAlgorithm{
        ChildAlgorithm <: Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: FlattenedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion =
        AI.StopAfterIteration(sum(max_iterations, algorithms))
end
function DefaultFlattenedAlgorithm(f::Function, nalgorithms::Int; kwargs...)
    return DefaultFlattenedAlgorithm(; algorithms = f.(1:nalgorithms), kwargs...)
end

@kwdef mutable struct DefaultFlattenedAlgorithmState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: FlattenedAlgorithmState
    iterate::Iterate
    iteration::Int = 0
    parent_iteration::Int = 1
    child_iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end #============================ NonIterativeAlgorithm =======================================#

# Algorithm that only performs a single step.
abstract type NonIterativeAlgorithm <: Algorithm end
abstract type NonIterativeAlgorithmState <: State end

function AI.initialize_state(problem::Problem, algorithm::NonIterativeAlgorithm; kwargs...)
    return DefaultNonIterativeAlgorithmState(; kwargs...)
end
function AI.solve!(
        problem::Problem, algorithm::NonIterativeAlgorithm, state::State; kwargs...
    )
    return throw(MethodError(AI.solve!, (problem, algorithm, state)))
end

@kwdef mutable struct DefaultNonIterativeAlgorithmState{Iterate} <:
    NonIterativeAlgorithmState
    iterate::Iterate
end

end
