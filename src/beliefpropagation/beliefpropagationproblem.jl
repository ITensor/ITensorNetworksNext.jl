import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using BackendSelection: @Algorithm_str, Algorithm
using DataGraphs: edge_data
using Graphs: AbstractEdge, edges, has_edge, vertices
using LinearAlgebra: norm, normalize
using NamedDimsArrays: AbstractNamedDimsArray
using NamedGraphs.GraphsExtensions: add_edges!, boundary_edges, subgraph
using NamedGraphs.PartitionedGraphs: quotientvertices

@kwdef struct StopWhenConverged <: AI.StoppingCriterion
    tol::Float64
end

@kwdef mutable struct StopWhenConvergedState{Iterate} <: AI.StoppingCriterionState
    delta::Float64 = Inf
    at_iteration::Int = -1
    previous_iterate::Iterate
end

function AI.initialize_state(::AIE.Problem, ::AIE.Algorithm, ::StopWhenConverged; iterate)
    return StopWhenConvergedState(; previous_iterate = copy(iterate))
end

function AI.initialize_state!(
        ::AIE.Problem,
        ::AIE.Algorithm,
        ::StopWhenConverged,
        st::StopWhenConvergedState
    )
    st.delta = Inf
    return st
end

function AI.is_finished!(
        problem::AIE.Problem,
        algorithm::AIE.Algorithm,
        state::AIE.State,
        c::StopWhenConverged,
        st::StopWhenConvergedState
    )
    iterate = state.iterate
    previous_iterate = st.previous_iterate

    delta = iterate_diff(iterate, previous_iterate)

    st.previous_iterate = copy(iterate)

    # maxdiff = 0.0 initially, so skip this the first time.
    state.iteration == 0 && return false

    st.delta = delta

    if AI.is_finished(problem, algorithm, state, c, st)
        st.at_iteration = state.iteration
        return true
    end

    return false
end

function AI.is_finished(
        ::AIE.Problem,
        ::AIE.Algorithm,
        ::AIE.State,
        c::StopWhenConverged,
        st::StopWhenConvergedState
    )
    return st.delta < c.tol
end

struct BeliefPropagationProblem{Factors} <: AIE.Problem
    factors::Factors
end

function iterate_diff(
        cache1::MessageCache,
        cache2::MessageCache
    )
    return maximum(edges(cache1)) do edge
        m1 = cache1[edge]
        m2 = cache2[edge]
        return 1 - abs2(LinearAlgebra.dot(normalize(m1), normalize(m2)))
    end
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

struct SimpleMessageUpdate{E <: AbstractEdge, Kwargs <: NamedTuple}
    edge::E
    kwargs::Kwargs
end

function SimpleMessageUpdate(
        edge;
        normalize = true,
        contraction_alg = Algorithm"exact",
        kwargs...
    )
    return SimpleMessageUpdate(
        edge,
        (; normalize, contraction_alg, kwargs...)
    )
end

function Base.getproperty(alg::SimpleMessageUpdate, name::Symbol)
    if name in (:edge, :kwargs)
        return getfield(alg, name)
    else
        return getproperty(getfield(alg, :kwargs), name)
    end
end

AI.initialize_state(::BeliefPropagationProblem, ::SimpleMessageUpdate; iterate) = iterate

struct BeliefPropagationSweep{
        ChildAlgorithm, Algorithms <: AbstractVector{ChildAlgorithm},
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::AI.StopAfterIteration
    function BeliefPropagationSweep(; algorithms)
        stopping_criterion = AI.StopAfterIteration(length(algorithms))
        return new{eltype(algorithms), typeof(algorithms)}(algorithms, stopping_criterion)
    end
end

function BeliefPropagationSweep(f::Function, edges)
    return BeliefPropagationSweep(; algorithms = f.(edges))
end

function AIE.set_substate!(
        ::BeliefPropagationProblem,
        ::BeliefPropagationSweep,
        state::AIE.DefaultState,
        cache::MessageCache
    )
    state.iterate = cache

    return state
end

function AI.solve!(
        problem::BeliefPropagationProblem,
        algorithm::SimpleMessageUpdate,
        cache::MessageCache;
        logging_context_prefix = AIE.default_logging_context_prefix(problem, algorithm)
    )
    edge = algorithm.edge

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

    return cache
end

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

    algorithm = BeliefPropagation(maxiter; stopping_criterion) do repnum
        return BeliefPropagationSweep(edges) do edge
            return SimpleMessageUpdate(edge; edge_kwargs[repnum]...)
        end
    end

    ##

    state = AI.solve(problem, algorithm; iterate = cache)

    return state.iterate # -> typeof(cache)
end
