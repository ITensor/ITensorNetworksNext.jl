using Dagger
using Dagger.Distributed

using DataGraphs: DataGraphs, get_edge_data, get_vertex_data, is_edge_assigned,
    is_vertex_assigned, set_edge_data!, set_vertex_data!, underlying_graph
using Dictionaries: Indices
using Graphs: AbstractEdge, AbstractGraph, dst, edges, src, vertices
using ITensorNetworksNext.ITensorNetworksNextParallel: DaggerBeliefPropagationCache,
    DaggerNestedAlgorithm, DaggerState, ITensorNetworksNextParallel, dagger_algorithm,
    subcache
using ITensorNetworksNext: BeliefPropagation, BeliefPropagationCache,
    BeliefPropagationProblem, BeliefPropagationState, ITensorNetworksNext,
    beliefpropagation, forest_cover_edge_sequence, select_algorithm
using NamedGraphs.PartitionedGraphs: QuotientVertex, quotientedges, quotientvertices
using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: boundary_edges

function ITensorNetworksNextParallel.subcache(cache::DaggerBeliefPropagationCache, inds)
    return subcache(cache.underlying_cache, inds)
end

function ITensorNetworksNextParallel.DaggerBeliefPropagationCache(network::AbstractGraph)
    underlying_cache = BeliefPropagationCache(network)

    keys = Indices(quotientvertices(underlying_cache))

    workers = Iterators.cycle(Distributed.workers())
    worker_dict = similar(keys, Int)

    for quotient_vertex in keys
        worker, workers = Iterators.peel(workers)
        worker_dict[quotient_vertex] = worker
    end

    quotient_chunks = map(keys) do quotient_vertex
        worker = worker_dict[quotient_vertex]
        iterate = subcache(underlying_cache, quotient_vertex)
        chunk = Dagger.@mutable worker = worker BeliefPropagationState(; iterate)
        return chunk
    end

    return DaggerBeliefPropagationCache(underlying_cache, quotient_chunks)
end

DataGraphs.underlying_graph(cache::DaggerBeliefPropagationCache) = underlying_graph(cache.underlying_cache)

DataGraphs.is_vertex_assigned(bpc::DaggerBeliefPropagationCache, vertex) = is_vertex_assigned(bpc.underlying_cache, vertex)
DataGraphs.is_edge_assigned(bpc::DaggerBeliefPropagationCache, edge) = is_edge_assigned(bpc.undelying_cache, edge)

DataGraphs.get_vertex_data(bpc::DaggerBeliefPropagationCache, vertex) = get_vertex_data(bpc.underlying_cache, vertex)
DataGraphs.get_edge_data(bpc::DaggerBeliefPropagationCache, edge::AbstractEdge) = get_edge_data(bpc.undelying_caches, edge)

DataGraphs.set_vertex_data!(bpc::DaggerBeliefPropagationCache, val, vertex) = set_vertex_data!(bpc.underlying_cache, val, vertex)
DataGraphs.set_edge_data!(bpc::DaggerBeliefPropagationCache, val, edge) = set_edge_data!(bpc.underlying_cache, val, edge)

NamedGraphs.to_graph_index(::DaggerBeliefPropagationCache, qv::QuotientVertex) = qv
function DataGraphs.get_index_data(cache::DaggerBeliefPropagationCache, qv::QuotientVertex)
    return cache.quotient_chunks[qv]
end

function ITensorNetworksNext.beliefpropagation_sweep(cache::DaggerBeliefPropagationCache; edges, workers = workers(), kwargs...)

    keys = collect(quotientvertices(cache))

    return dagger_algorithm(keys; keys, workers) do quotient_vertex

        subcache = fetch(cache[quotient_vertex]).iterate

        subcache_edges = forest_cover_edge_sequence(subcache) ∩ edges
        incoming_edges = boundary_edges(cache, vertices(cache, quotient_vertex); dir = :in)

        alg = select_algorithm(
            beliefpropagation,
            subcache;
            # Don't update the incoming messages
            edges = setdiff(subcache_edges, incoming_edges),
            maxiter = 1,
            kwargs...
        )

        return alg
    end
end

function AI.initialize_state(
        problem::AIE.Problem,
        algorithm::BeliefPropagation{<:DaggerNestedAlgorithm};
        kwargs...
    )
    return initialize_dagger_state(problem, algorithm; kwargs...)
end

function AIE.get_subproblem(
        problem::BeliefPropagationProblem,
        algorithm::DaggerNestedAlgorithm,
        state::DaggerState,
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]

    quotient_vertex = algorithm.keys[state.iteration]

    cache = state.iterate.iterate

    subiterate = cache[quotient_vertex]

    return subproblem, subalgorithm, subiterate
end

function AIE.set_substate!(
        ::BeliefPropagationProblem,
        algorithm::AIE.NestedAlgorithm,
        state::AIE.State,
        substate::DaggerState,
    )

    dst_cache = state.iterate.iterate

    state.iterate.maxdiff = 0.0

    current_algorithm = algorithm.algorithms[state.iteration]

    for (i, quotient_vertex) in enumerate(current_algorithm.keys)
        get_maxdiff = dtask -> dtask.iterate.maxdiff
        src_maxdiff = fetch(Dagger.@spawn get_maxdiff(substate.remote_results[i]))

        if src_maxdiff > state.iterate.maxdiff
            state.iterate.maxdiff = src_maxdiff
        end
    end


    transfer_edges! = (dst_chunk, src_chunk, edges) -> begin
        src_subcache = src_chunk.iterate
        dst_subcache = dst_chunk.iterate
        for edge in edges
            dst_subcache[edge] = src_subcache[edge]
        end
    end

    transfer_dtasks = map(quotientedges(dst_cache)) do quotient_edge
        src_subcache = dst_cache[src(quotient_edge)]
        dst_subcache = dst_cache[dst(quotient_edge)]
        return Dagger.@spawn transfer_edges!(dst_subcache, fetch(src_subcache), edges(dst_cache, quotient_edge))
    end

    wait.(transfer_dtasks)

    return state
end
