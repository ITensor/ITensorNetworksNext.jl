using Graphs: AbstractEdge, edges, has_edge, vertices
using NamedGraphs.GraphsExtensions: add_edges!, boundary_edges, subgraph
using NamedDimsArrays: AbstractNamedDimsArray
using NamedGraphs.PartitionedGraphs: quotientvertices
using DataGraphs: edge_data
using LinearAlgebra: norm, normalize

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
    end

    return st.delta < c.tol
end

struct BeliefPropagationProblem{Network} <: AIE.Problem
    network::Network
end

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
        contraction_alg = "exact",
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
        ::BeliefPropagationProblem, ::AIE.NonIterativeAlgorithm; iterate, kwargs...
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
        ::BeliefPropagationSweep,
        sweep_state::AIE.DefaultState,
        noniterative_substate::BeliefPropagationState,
    )

    sweep_state.iterate = noniterative_substate

    return sweep_state
end

struct MessageUpdateProblem{Factor, Messages} <: AIE.Problem
    factor::Factor
    messages::Messages
end

function AI.solve!(
        problem::BeliefPropagationProblem,
        algorithm::SimpleMessageUpdate,
        state::BeliefPropagationState;
        logging_context_prefix = AIE.default_logging_context_prefix(problem, algorithm),
    )

    logger = AI.algorithm_logger()

    cache = state.iterate
    edge = algorithm.edge

    AI.emit_message(
        logger, problem, algorithm, state, Symbol(logging_context_prefix, :PreUpdate)
    )

    new_message = updated_message(algorithm, cache)

    if !isnothing(algorithm.message_diff_function)
        diff = algorithm.message_diff_function(new_message, cache[edge])

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

default_message_diff_function(m1, m2) = norm(normalize(m1) - normalize(m2))

function updated_message(algorithm, cache)
    edge = algorithm.edge

    vertex = src(edge)
    messages = incoming_messages(cache, vertex; ignore_edges = typeof(edge)[reverse(edge)])

    update_problem = MessageUpdateProblem(cache[vertex], messages)

    message_state = AI.solve(update_problem, algorithm; iterate = message(cache, edge))

    return message_state.iterate
end

function AI.solve!(
        problem::MessageUpdateProblem,
        algorithm::SimpleMessageUpdate,
        state::AIE.DefaultNonIterativeAlgorithmState;
        logging_context_prefix = AIE.default_logging_context_prefix(problem, algorithm),
        kwargs...
    )

    logger = AI.algorithm_logger()

    AI.emit_message(
        logger, problem, algorithm, state, Symbol(logging_context_prefix, :PreUpdate)
    )

    state.iterate = contract_messages(algorithm.contraction_alg, problem.factor, problem.messages)

    AI.emit_message(
        logger, problem, algorithm, state, Symbol(logging_context_prefix, :PreNormalization)
    )

    if algorithm.normalize
        # TODO: use `sum` not `norm`
        message_norm = LinearAlgebra.norm(state.iterate)
        if !iszero(message_norm)
            state.iterate /= message_norm
        end
    end

    AI.emit_message(
        logger, problem, algorithm, state, Symbol(logging_context_prefix, :PostNormalization)
    )

    return state
end

function contract_messages(alg, factor::AbstractArray, messages)
    factors = typeof(factor)[factor]
    return contract_network(vcat(factors, messages); alg)
end

beliefpropagation(network; kwargs...) = beliefpropagation(BeliefPropagationCache(network), network; kwargs...)
function beliefpropagation(cache::AbstractBeliefPropagationCache, network = nothing; kwargs...)

    problem = BeliefPropagationProblem(network)

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
        edges = forest_cover_edge_sequence(cache),
        maxiter = is_tree(cache) ? 1 : nothing,
        tol = -Inf,
        message_diff_function = tol > -Inf ? (m1, m2) -> norm(m1 / norm(m1) - m2 / norm(m2)) : nothing,
        kwargs...
    )

    if isnothing(maxiter)
        throw(ArgumentError("`maxiter` must be specified for non-tree graphs"))
    end

    stopping_criterion = AI.StopAfterIteration(maxiter)

    if tol > -Inf
        stopping_criterion = stopping_criterion | StopWhenConverged(tol)
    end

    extended_kwargs = extend_columns((; message_diff_function, kwargs...), maxiter)
    edge_kwargs = rows(extended_kwargs, maxiter)

    return BeliefPropagation(maxiter; stopping_criterion) do repnum
        return beliefpropagation_sweep(cache; edges, edge_kwargs[repnum]...)
    end
end

# A single sweep across the given edges.
function beliefpropagation_sweep(::BeliefPropagationCache; edges, kwargs...)
    return BeliefPropagationSweep(edges) do edge
        return SimpleMessageUpdate(edge; kwargs...)
    end
end
