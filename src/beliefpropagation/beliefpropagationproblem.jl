using Graphs: SimpleGraph, vertices, edges, has_edge, AbstractEdge
using NamedGraphs: AbstractNamedGraph, position_graph
using NamedGraphs.GraphsExtensions: add_edges!, partition_vertices, subgraph, boundary_edges
using NamedGraphs.OrderedDictionaries: OrderedDictionary, OrderedIndices
using NamedDimsArrays: AbstractNamedDimsArray
using NamedGraphs.PartitionedGraphs: quotientvertices
using DataGraphs: edge_data

import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

@kwdef struct StopWhenConverged <: AI.StoppingCriterion
    tol::Float64 = 0.0
end

@kwdef mutable struct StopWhenConvergedState <: AI.StoppingCriterionState
    delta::Float64 = Inf
end

function AI.initialize_state(::AIE.Problem, ::AIE.Algorithm, ::StopWhenConverged)
    return StopWhenConvergedState()
end

function AI.initialize_state!(
        ::AIE.Problem,
        ::AIE.Algorithm,
        ::StopWhenConverged,
        st::StopWhenConvergedState,
    )
    st.delta = Inf
    return st
end

function AI.is_finished!(
        ::AIE.Problem,
        ::AIE.Algorithm,
        state::AIE.State,
        c::StopWhenConverged,
        st::StopWhenConvergedState,
    )

    # maxdiff = 0.0 initially, so skip this the first time.
    if state.iteration > 0
        st.delta = state.iterate.maxdiff
        @info "$(state.iteration): $(st.delta)"
    end

    return st.delta < c.tol
end

# struct BeliefPropagationProblem{Network} <: AIE.Problem
#     network::Network
# end

struct BeliefPropagationProblem <: AIE.Problem end

@kwdef mutable struct BeliefPropagationState{Iterate, Diffs} <: AIE.NonIterativeAlgorithmState
    iterate::Iterate
    diffs::Diffs = similar(edge_data(iterate), Float64)
    maxdiff::Float64 = 0.0
end

@kwdef struct BeliefPropagation{
        ChildAlgorithm <: AIE.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end

function BeliefPropagation(f::Function, niterations::Int; kwargs...)
    return BeliefPropagation(; algorithms = f.(1:niterations), kwargs...)
end

abstract type AbstractMessageUpdate <: AIE.NonIterativeAlgorithm end

struct SimpleMessageUpdate{E <: AbstractEdge, Kwargs <: NamedTuple} <: AbstractMessageUpdate
    edge::E
    kwargs::Kwargs
end

function SimpleMessageUpdate(
        edge;
        normalize = true,
        contraction_alg = "eager",
        compute_diff = false,
        kwargs...
    )
    return SimpleMessageUpdate(edge, (; normalize, contraction_alg, compute_diff, kwargs...))
end

function Base.getproperty(alg::SimpleMessageUpdate, name::Symbol)
    if name in (:edge, :kwargs)
        return getfield(alg, name)
    else
        return getproperty(getfield(alg, :kwargs), name)
    end
end

struct BeliefPropagationSweep{
        ChildAlgorithm <: AIE.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::AI.StopAfterIteration
    function BeliefPropagationSweep(; algorithms)
        stopping_criterion = AI.StopAfterIteration(length(algorithms))
        return new{eltype(algorithms), typeof(algorithms)}(algorithms, stopping_criterion)
    end
end

BeliefPropagationSweep(f::Function, edges) = BeliefPropagationSweep(; algorithms = f.(edges))

function AI.initialize_state(
        problem::BeliefPropagationProblem,
        update_algorithm::AIE.NonIterativeAlgorithm; iterate, kwargs...
    )

    diffs = iterate.diffs
    maxdiff = iterate.maxdiff

    return BeliefPropagationState(; iterate = iterate.iterate, diffs, maxdiff, kwargs...)
end

# This gets called at the start of every sweep.
function AI.initialize_state!(
        ::BeliefPropagationProblem,
        ::BeliefPropagationSweep,
        iteration_state::AIE.State,
    )
    iteration_state.iterate.maxdiff = 0.0
    return iteration_state
end

function AIE.set_substate!(
        ::BeliefPropagationProblem,
        sweep_algorithm::BeliefPropagationSweep,
        sweep_state::AIE.DefaultState,
        noniterative_substate::BeliefPropagationState,
    )

    sweep_state.iterate = noniterative_substate

    return sweep_state
end

struct MessageUpdateProblem{Messages, Factors} <: AIE.Problem
    messages::Messages
    factors::Factors
end

function AI.solve!(
        problem::BeliefPropagationProblem,
        algorithm::SimpleMessageUpdate,
        state::BeliefPropagationState;
        logging_context_prefix = default_logging_context_prefix(problem, algorithm),
    )

    logger = AI.algorithm_logger()

    cache = state.iterate
    edge = algorithm.edge

    AI.emit_message(
        logger, problem, algorithm, state, Symbol(logging_context_prefix, :PreUpdate)
    )

    new_message = updated_message(algorithm, cache)

    if algorithm.compute_diff
        diff = message_diff(new_message, cache[edge])

        if diff > state.maxdiff
            state.maxdiff = diff
        end

        state.diffs[edge] = diff
    end

    setmessage!(cache, edge, new_message)

    AI.emit_message(
        logger, problem, algorithm, state, Symbol(logging_context_prefix, :PostUpdate)
    )

    return state
end

message_diff(m1, m2) = LinearAlgebra.norm(m1 - m2)

function updated_message(algorithm, cache)
    edge = algorithm.edge

    vertex = src(edge)
    messages = incoming_messages(cache, vertex; ignore_edges = typeof(edge)[reverse(edge)])

    update_problem = MessageUpdateProblem(messages, factors(cache, vertex))

    message_state = AI.solve(update_problem, algorithm; iterate = message(cache, edge))

    return message_state.iterate
end

function AI.solve!(
        problem::MessageUpdateProblem,
        algorithm::SimpleMessageUpdate,
        state::AIE.DefaultNonIterativeAlgorithmState;
        logging_context_prefix = AI.default_logging_context_prefix(problem, algorithm),
        kwargs...
    )

    # TODO: logging...

    state.iterate = contract_messages(algorithm.contraction_alg, problem.factors, problem.messages)

    if algorithm.normalize
        # TODO: use `sum` not `norm`
        message_norm = LinearAlgebra.norm(state.iterate)
        if !iszero(message_norm)
            state.iterate /= message_norm
        end
    end

    return state
end

contract_messages(alg, factors, messages) = not_implemented()
function contract_messages(
        alg,
        factors::Vector{<:AbstractArray},
        messages::Vector{<:AbstractArray},
    )
    return contract_network(vcat(factors, messages); alg)
end

beliefpropagation(network; kwargs...) = beliefpropagation(BeliefPropagationCache(network); kwargs...)
function beliefpropagation(cache::AbstractBeliefPropagationCache; kwargs...)

    # problem = BeliefPropagationProblem(network(cache))
    problem = BeliefPropagationProblem()

    algorithm = select_algorithm(beliefpropagation, cache; kwargs...)

    # The nested algorithms will wrap and manipulate this object.

    base_state = BeliefPropagationState(; iterate = cache)

    state = AI.initialize_state(problem, algorithm; iterate = base_state)

    state = AI.solve!(problem, algorithm, state)

    return state.iterate.iterate
end


function select_algorithm(
        ::typeof(beliefpropagation),
        cache::AbstractBeliefPropagationCache;
        edges = forest_cover_edge_sequence(network(cache)),
        maxiter = is_tree(network(cache)) ? 1 : nothing,
        tol = -Inf,
        kwargs...
    )

    if isnothing(maxiter)
        throw(ArgumentError("`maxiter` must be specified for non-tree graphs"))
    end

    stopping_criterion = AI.StopAfterIteration(maxiter)
    compute_diff = false

    if tol > -Inf
        stopping_criterion = stopping_criterion | StopWhenConverged(tol)
        compute_diff = true
    end

    extended_kwargs = extend_columns((; compute_diff, kwargs...), maxiter)
    edge_kwargs = rows(extended_kwargs, len = maxiter)

    return BeliefPropagation(maxiter; stopping_criterion) do repnum
        return beliefpropagation_sweep(cache; edges, edge_kwargs[repnum]...)
    end
end

# A single sweep across the given edges.
function beliefpropagation_sweep(cache::BeliefPropagationCache; edges, kwargs...)
    return BeliefPropagationSweep(edges) do edge
        return SimpleMessageUpdate(edge; kwargs...)
    end
end
