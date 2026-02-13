using DataGraphs: DataGraphs, get_edge_data, get_vertex_data, is_edge_assigned,
    is_vertex_assigned, set_edge_data!, set_vertex_data!, underlying_graph
using Graphs: AbstractEdge, AbstractGraph, edges, vertices
using ITensorNetworksNext.ITensorNetworksNextParallel: DistributedBeliefPropagationCache,
    DistributedNestedAlgorithm, DistributedState, ITensorNetworksNextParallel,
    distributed_algorithm
using ITensorNetworksNext: BeliefPropagation, BeliefPropagationCache,
    BeliefPropagationProblem, BeliefPropagationState, ITensorNetworksNext,
    beliefpropagation, forest_cover_edge_sequence, select_algorithm, setmessages!
using NamedGraphs.GraphsExtensions: boundary_edges
using NamedGraphs.PartitionedGraphs: QuotientVertex, quotientvertices
using NamedGraphs: NamedGraphs

function ITensorNetworksNextParallel.DistributedBeliefPropagationCache(network::AbstractGraph)
    underlying_cache = BeliefPropagationCache(network)
    return DistributedBeliefPropagationCache(underlying_cache)
end

DataGraphs.underlying_graph(cache::DistributedBeliefPropagationCache) = underlying_graph(cache.underlying_cache)

DataGraphs.is_vertex_assigned(bpc::DistributedBeliefPropagationCache, vertex) = is_vertex_assigned(bpc.underlying_cache, vertex)
DataGraphs.is_edge_assigned(bpc::DistributedBeliefPropagationCache, edge) = is_edge_assigned(bpc.undelying_cache, edge)

DataGraphs.get_vertex_data(bpc::DistributedBeliefPropagationCache, vertex) = get_vertex_data(bpc.underlying_cache, vertex)
DataGraphs.get_edge_data(bpc::DistributedBeliefPropagationCache, edge::AbstractEdge) = get_edge_data(bpc.underlying_cache, edge)

DataGraphs.set_vertex_data!(bpc::DistributedBeliefPropagationCache, val, vertex) = set_vertex_data!(bpc.underlying_cache, val, vertex)
DataGraphs.set_edge_data!(bpc::DistributedBeliefPropagationCache, val, edge) = set_edge_data!(bpc.underlying_cache, val, edge)

NamedGraphs.to_graph_index(::DistributedBeliefPropagationCache, qv::QuotientVertex) = qv
function DataGraphs.get_index_data(cache::DistributedBeliefPropagationCache, qv::QuotientVertex)
    return ITensorNetworksNextParallel.subcache(cache.underlying_cache, qv)
end
function ITensorNetworksNext.beliefpropagation_sweep(
        cache::DistributedBeliefPropagationCache; edges, kwargs...
    )

    keys = collect(quotientvertices(cache))

    return distributed_algorithm(keys; keys, workers = WorkerPool(workers())) do quotient_vertex

        subcache = cache[quotient_vertex]
        subcache_edges = forest_cover_edge_sequence(subcache) ∩ edges
        incoming_edges = boundary_edges(cache, vertices(cache, quotient_vertex); dir = :in)

        alg = select_algorithm(
            beliefpropagation,
            subcache;
            edges = setdiff(subcache_edges, incoming_edges),
            maxiter = 1,
            kwargs...
        )

        return alg
    end
end

function AI.initialize_state(
        problem::AIE.Problem,
        algorithm::BeliefPropagation{<:DistributedNestedAlgorithm};
        kwargs...
    )

    keys = first(algorithm.algorithms).keys

    return initialize_distributed_state(problem, algorithm; keys = keys, kwargs...)
end

function AIE.get_subproblem(
        problem::BeliefPropagationProblem,
        algorithm::DistributedNestedAlgorithm,
        state::DistributedState
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]

    cache = state.iterate.iterate

    quotient_vertex = algorithm.keys[state.iteration]
    subiterate = BeliefPropagationState(; iterate = cache[quotient_vertex])

    return subproblem, subalgorithm, subiterate
end

function AIE.set_substate!(
        ::BeliefPropagationProblem,
        ::AIE.NestedAlgorithm,
        state::AIE.State,
        substate::DistributedState,
    )

    dst_cache = state.iterate.iterate

    state.iterate.maxdiff = 0.0

    for quotient_vertex in quotientvertices(dst_cache)

        src_state = fetch(substate.remote_results[quotient_vertex]).iterate

        src_cache = src_state.iterate
        src_maxdiff = src_state.maxdiff

        incoming_edges = boundary_edges(dst_cache, vertices(dst_cache, quotient_vertex); dir = :in)

        updated_messages = setdiff(edges(src_cache), incoming_edges)

        setmessages!(dst_cache, src_cache, updated_messages)

        if src_maxdiff > state.iterate.maxdiff
            state.iterate.maxdiff = src_maxdiff
        end

    end

    return state
end
