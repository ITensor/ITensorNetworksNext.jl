import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using BackendSelection: @Algorithm_str, Algorithm
using DataGraphs: edge_data
using Graphs: AbstractEdge, edges, has_edge, vertices
using LinearAlgebra: norm, normalize
using NamedDimsArrays: AbstractNamedDimsArray
using NamedGraphs.GraphsExtensions: add_edges!, boundary_edges, subgraph
using NamedGraphs.PartitionedGraphs: quotientvertices

@kwdef struct StopWhenConverged{Tol <: Real} <: AI.StoppingCriterion
    tol::Tol = NaN
end

@kwdef mutable struct StopWhenConvergedState{Iterate, Delta <: Real} <:
    AI.StoppingCriterionState
    delta::Delta = NaN
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
    st.delta = NaN
    return st
end

function AI.is_finished!(
        problem::AIE.Problem,
        algorithm::AIE.Algorithm,
        state::AIE.State,
        c::StopWhenConverged,
        st::StopWhenConvergedState
    )

    # maxdiff = 0.0 initially, so skip this the first time.
    iterate = state.iterate
    previous_iterate = st.previous_iterate

    delta = iterate_diff(iterate, previous_iterate)

    st.previous_iterate = copy(iterate)

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

struct BeliefPropagationProblem{Network} <: AIE.Problem
    network::Network
end

function iterate_diff(
        cache1::AbstractBeliefPropagationCache,
        cache2::AbstractBeliefPropagationCache
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
        cache::AbstractBeliefPropagationCache
    )
    state.iterate = cache

    return state
end

function AI.solve!(
        ::BeliefPropagationProblem,
        algorithm::SimpleMessageUpdate,
        cache::AbstractBeliefPropagationCache; kwargs...
    )
    edge = algorithm.edge

    vertex = src(edge)
    messages = incoming_messages(cache, vertex; ignore_edges = [reverse(edge)])

    tensors = vcat([factor(cache, vertex)], messages)

    new_message = contract_network(tensors; algorithm.contraction_alg)

    if algorithm.normalize
        message_norm = sum(new_message)
        if !iszero(message_norm)
            new_message /= message_norm
        end
    end

    setmessage!(cache, edge, new_message)

    return cache
end

function beliefpropagation(network; kwargs...)
    return beliefpropagation(BeliefPropagationCache(network), network; kwargs...)
end

function beliefpropagation(
        cache::AbstractBeliefPropagationCache,
        network = nothing;
        kwargs...
    )
    problem = BeliefPropagationProblem(network)

    algorithm = select_algorithm(beliefpropagation, cache; kwargs...)

    state = AI.solve(problem, algorithm; iterate = cache)

    return state.iterate
end

function select_algorithm(
        ::typeof(beliefpropagation),
        cache::AbstractBeliefPropagationCache;
        edges = forest_cover_edge_sequence(cache),
        maxiter = is_tree(cache) ? 1 : nothing,
        tol = NaN,
        kwargs...
    )
    if isnothing(maxiter)
        throw(ArgumentError("`maxiter` must be specified for non-tree graphs"))
    end

    stopping_criterion = AI.StopAfterIteration(maxiter)

    if !isnan(tol)
        stopping_criterion = stopping_criterion | StopWhenConverged(tol)
    end

    extended_kwargs = extend_columns((; kwargs...), maxiter)
    edge_kwargs = rows(extended_kwargs, maxiter)

    return BeliefPropagation(maxiter; stopping_criterion) do repnum
        return BeliefPropagationSweep(edges) do edge
            return SimpleMessageUpdate(edge; edge_kwargs[repnum]...)
        end
    end
end
