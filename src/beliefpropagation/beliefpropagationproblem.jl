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

function beliefpropagation(
        factors, messages;
        edges = nothing,
        maxiter = is_tree(factors) ? 1 : nothing,
        stopping_criterion = nothing,
        kwargs...
    )
    if isnothing(maxiter)
        throw(
            ArgumentError(
                "`maxiter` must be specified for non-tree graphs, even when
                    `stopping_criterion` is provided."
            )
        )
    end

    cache = MessageCache(messages)
    problem = BeliefPropagationProblem(factors)

    ## Algorithm construction:

    edges = isnothing(edges) ? forest_cover_edge_sequence(cache) : edges

    base_stopping_criterion = AI.StopAfterIteration(maxiter)

    if !isnothing(stopping_criterion)
        base_stopping_criterion |= stopping_criterion
    end

    stopping_criterion = base_stopping_criterion

    extended_kwargs = extend_columns((; kwargs...), maxiter)
    edge_kwargs = rows(extended_kwargs, maxiter)

    algorithm = BeliefPropagationAlgorithm(maxiter; edges, stopping_criterion) do repnum
        message_update_algorithm = SimpleMessageUpdateAlgorithm(;
            edge_kwargs[repnum]...
        )
        return BeliefPropagationSweepAlgorithm(;
            algorithms = Dict(edge => message_update_algorithm for edge in edges)
        )
    end

    ##

    return AI.solve(problem, algorithm; iterate = cache) # -> typeof(cache)
end

# === Layer 1: BP outer loop (iterative) ===

struct BeliefPropagationProblem{Factors} <: AI.Problem
    factors::Factors
end

@kwdef struct BeliefPropagationAlgorithm{
        Edges,
        Algorithms,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    edges::Edges
    # Indexable by iteration count (e.g. `Vector` or `Dict{Int, ...}`).
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end

function BeliefPropagationAlgorithm(f::Function, niterations::Int; edges, kwargs...)
    return BeliefPropagationAlgorithm(; edges, algorithms = f.(1:niterations), kwargs...)
end

@kwdef mutable struct BeliefPropagationState{
        Iterate, SCState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::SCState
end

function AI.initialize_state(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return BeliefPropagationState(;
        iterate, iteration, stopping_criterion_state
    )
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
    subalgorithm = algorithm.algorithms[state.iteration]
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
        Algorithms,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    # Indexable by edge (e.g. `Dict{Edge, MessageUpdateAlgorithm}`); the
    # default constructor in `beliefpropagation()` builds one with the same
    # template copied across every edge.
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end

@kwdef mutable struct BeliefPropagationSweepState{
        Iterate, SCState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::SCState
end

function AI.initialize_state(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return BeliefPropagationSweepState(;
        iterate, iteration, stopping_criterion_state
    )
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
    subalgorithm = algorithm.algorithms[edge]
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

    new_message = contract_network(vcat(messages, [factor]); algorithm.contraction_alg)

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

# === Utility functions for processing keyword arguments ===

function repeat_last(v::AbstractVector, len::Int)
    return [v; fill(v[end], max(len - length(v), 0))]
end
repeat_last(v, len::Int) = fill(v, len)
function extend_columns(nt::NamedTuple, len::Int)
    return (; (keys(nt) .=> map(v -> repeat_last(v, len), values(nt)))...)
end
rowlength(nt::NamedTuple) = only(unique(length.(values(nt))))
function rows(nt::NamedTuple, len::Int = rowlength(nt))
    return [(; (keys(nt) .=> map(v -> v[i], values(nt)))...) for i in 1:len]
end
