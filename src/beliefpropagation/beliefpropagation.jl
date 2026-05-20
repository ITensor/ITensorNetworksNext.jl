import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using .AlgorithmsInterfaceExtensions: StopWhenConverged, iterate_diff
using BackendSelection: @Algorithm_str, Algorithm
using DataGraphs: edge_data
using Graphs: AbstractEdge, edges, has_edge, vertices
using LinearAlgebra: norm, normalize
using NamedDimsArrays: AbstractNamedDimsArray
using NamedGraphs.GraphsExtensions: add_edges!, boundary_edges, subgraph
using NamedGraphs.PartitionedGraphs: quotientvertices

# === Top-level user entry point ===

default_beliefpropagation_edges(graph) = forest_cover_edge_sequence(graph)

default_message_update_algorithm(; kwargs...) = SimpleMessageUpdateAlgorithm(; kwargs...)

select_message_update_algorithm(algorithm::AI.Algorithm) = algorithm
function select_message_update_algorithm(kwargs::NamedTuple)
    return default_message_update_algorithm(; kwargs...)
end

select_beliefpropagation_stopping_criterion(c::AI.StoppingCriterion) = c
function select_beliefpropagation_stopping_criterion(::Nothing)
    return throw(
        ArgumentError(
            "`stopping_criterion` must be specified, e.g.\n" *
                "  `stopping_criterion = (; maxiter = 10)` or\n" *
                "  `stopping_criterion = AI.StopAfterIteration(10) | StopWhenConverged(1.0e-10)`."
        )
    )
end
function select_beliefpropagation_stopping_criterion(kwargs::NamedTuple)
    return select_beliefpropagation_stopping_criterion(; kwargs...)
end
function select_beliefpropagation_stopping_criterion(; maxiter = nothing, kwargs...)
    if isnothing(maxiter)
        throw(ArgumentError("`maxiter` must be specified in `stopping_criterion`."))
    end
    if !isempty(kwargs)
        throw(
            ArgumentError(
                "Unrecognized `stopping_criterion` kwargs: $(keys(kwargs)). " *
                    "Only `maxiter` is currently supported."
            )
        )
    end
    return AI.StopAfterIteration(maxiter)
end

function beliefpropagation(
        factors, messages;
        edges = default_beliefpropagation_edges(factors),
        stopping_criterion = nothing,
        message_update_algorithm = default_message_update_algorithm()
    )
    problem = BeliefPropagationProblem(factors)

    message_update_algorithm = select_message_update_algorithm(message_update_algorithm)
    sweep_algorithm = BeliefPropagationSweepAlgorithm(;
        message_update_algorithm,
        stopping_criterion = AI.StopAfterIteration(length(edges))
    )
    stopping_criterion = select_beliefpropagation_stopping_criterion(stopping_criterion)
    algorithm = BeliefPropagationAlgorithm(; edges, sweep_algorithm, stopping_criterion)

    cache = MessageCache(messages)

    return AI.solve(problem, algorithm; iterate = cache) # -> typeof(cache)
end

# === Layer 1: BP outer loop (iterative) ===

struct BeliefPropagationProblem{Factors} <: AI.Problem
    factors::Factors
end

@kwdef struct BeliefPropagationAlgorithm{
        Edges,
        SweepAlgorithm <: AI.Algorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    edges::Edges
    sweep_algorithm::SweepAlgorithm
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct BeliefPropagationState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return BeliefPropagationState(; iterate, iteration, stopping_criterion_state)
end

function AI.initialize_state!(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm,
        state::BeliefPropagationState;
        iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AIE.initialize_subsolve(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm,
        state::BeliefPropagationState
    )
    subalgorithm = algorithm.sweep_algorithm
    subproblem = BeliefPropagationSweepProblem(problem.factors, algorithm.edges)
    substate = AI.initialize_state(subproblem, subalgorithm; state.iterate)
    return subproblem, subalgorithm, substate
end

# === Layer 2: one sweep over edges (iterative) ===

struct BeliefPropagationSweepProblem{Factors, Edges} <: AI.Problem
    factors::Factors
    edges::Edges
end

@kwdef struct BeliefPropagationSweepAlgorithm{
        MessageUpdateAlgorithm <: AI.Algorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    message_update_algorithm::MessageUpdateAlgorithm
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct BeliefPropagationSweepState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return BeliefPropagationSweepState(; iterate, iteration, stopping_criterion_state)
end

function AI.initialize_state!(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm,
        state::BeliefPropagationSweepState;
        iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AIE.initialize_subsolve(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm,
        state::BeliefPropagationSweepState
    )
    edge = problem.edges[state.iteration]
    subproblem = MessageUpdateProblem(problem.factors, edge)
    subalgorithm = algorithm.message_update_algorithm
    substate = AI.initialize_state(subproblem, subalgorithm; state.iterate)
    return subproblem, subalgorithm, substate
end

# === Layer 3: single-edge message update (non-iterative) ===

struct MessageUpdateProblem{Factors, Edge <: AbstractEdge} <: AI.Problem
    factors::Factors
    edge::Edge
end

@kwdef struct SimpleMessageUpdateAlgorithm{ContractionAlg} <: AI.Algorithm
    normalize::Bool = true
    contraction_alg::ContractionAlg = Algorithm"exact"
end

@kwdef mutable struct MessageUpdateState{Iterate} <: AI.State
    iterate::Iterate
end

function AI.initialize_state(
        ::MessageUpdateProblem, ::SimpleMessageUpdateAlgorithm; iterate
    )
    return MessageUpdateState(; iterate)
end

# Non-iterative algorithm: no per-call state to reset.
function AI.initialize_state!(
        ::MessageUpdateProblem, ::SimpleMessageUpdateAlgorithm, state::MessageUpdateState
    )
    return state
end

# Non-iterative algorithm: bypass the step!/stopping-criterion loop.
function AI.solve_loop!(
        problem::MessageUpdateProblem,
        algorithm::SimpleMessageUpdateAlgorithm,
        state::MessageUpdateState
    )
    cache = state.iterate
    edge = problem.edge

    messages = collect(incoming_messages(cache, edge))
    factor = problem.factors[src(edge)]

    new_message = contract_network([messages; [factor]]; algorithm.contraction_alg)

    if algorithm.normalize
        message_norm = sum(new_message)
        if !iszero(message_norm)
            new_message /= message_norm
        end
    end

    cache[edge] = new_message

    return state
end

# === `iterate_diff` for `MessageCache` (used by `AIE.StopWhenConverged`) ===

function AIE.iterate_diff(cache1::MessageCache, cache2::MessageCache)
    return maximum(edges(cache1)) do edge
        m1 = cache1[edge]
        m2 = cache2[edge]
        return 1 - abs2(LinearAlgebra.dot(normalize(m1), normalize(m2)))
    end
end
