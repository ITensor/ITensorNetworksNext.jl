import ITensorNetworksNext.ITensorNetworksNextParallel as ITNNP
using Dagger
using DataGraphs: DataGraphs, get_edge_data, get_vertex_data, is_edge_assigned,
    is_vertex_assigned, set_edge_data!, set_vertex_data!, underlying_graph
using Dictionaries: Indices
using Graphs: AbstractEdge, AbstractGraph, dst, edges, src, vertices
using ITensorNetworksNext: ITensorNetworksNext, BeliefPropagation, BeliefPropagationCache,
    BeliefPropagationProblem, BeliefPropagationState, beliefpropagation,
    forest_cover_edge_sequence, select_algorithm
using NamedGraphs.GraphsExtensions: boundary_edges
using NamedGraphs.PartitionedGraphs: QuotientVertex, quotientedges, quotientvertices
using NamedGraphs: NamedGraphs

function ITNNP.DaggerBeliefPropagationCache(network::AbstractGraph)
    underlying_cache = BeliefPropagationCache(network)

    keys = Indices(quotientvertices(underlying_cache))

    workers = Iterators.cycle(Dagger.Distributed.workers())
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

    return ITNNP.DaggerBeliefPropagationCache(underlying_cache, quotient_chunks)
end

function DataGraphs.underlying_graph(cache::ITNNP.DaggerBeliefPropagationCache)
    return underlying_graph(cache.underlying_cache)
end

function DataGraphs.is_vertex_assigned(bpc::ITNNP.DaggerBeliefPropagationCache, vertex)
    return is_vertex_assigned(bpc.underlying_cache, vertex)
end
function DataGraphs.is_edge_assigned(bpc::ITNNP.DaggerBeliefPropagationCache, edge)
    return is_edge_assigned(bpc.undelying_cache, edge)
end

function DataGraphs.get_vertex_data(bpc::ITNNP.DaggerBeliefPropagationCache, vertex)
    return get_vertex_data(bpc.underlying_cache, vertex)
end
function DataGraphs.get_edge_data(
        bpc::ITNNP.DaggerBeliefPropagationCache,
        edge::AbstractEdge
    )
    return get_edge_data(bpc.undelying_caches, edge)
end

function DataGraphs.set_vertex_data!(bpc::ITNNP.DaggerBeliefPropagationCache, val, vertex)
    return set_vertex_data!(bpc.underlying_cache, val, vertex)
end
function DataGraphs.set_edge_data!(bpc::ITNNP.DaggerBeliefPropagationCache, val, edge)
    return set_edge_data!(bpc.underlying_cache, val, edge)
end

NamedGraphs.to_graph_index(::ITNNP.DaggerBeliefPropagationCache, qv::QuotientVertex) = qv
function DataGraphs.get_index_data(
        cache::ITNNP.DaggerBeliefPropagationCache,
        qv::QuotientVertex
    )
    return cache.quotient_chunks[qv]
end

function ITensorNetworksNext.beliefpropagation_sweep(
        cache::ITNNP.DaggerBeliefPropagationCache;
        edges,
        workers = workers(),
        kwargs...
    )
    keys = collect(quotientvertices(cache))

    return ITNNP.dagger_algorithm(keys; keys, workers) do quotient_vertex
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
        algorithm::BeliefPropagation{<:ITNNP.DaggerNestedAlgorithm};
        kwargs...
    )
    return ITNNP.initialize_dagger_state(problem, algorithm; kwargs...)
end

function ITNNP.get_subiterate(
        ::BeliefPropagationProblem,
        ::BeliefPropagation, # Our parallel region runs a small BP
        state::ITNNP.DaggerState
    )
    cache = state.iterate.iterate

    quotient_vertex = collect(quotientvertices(cache))[state.iteration]

    subiterate = cache[quotient_vertex]

    return subiterate
end

function AIE.set_substate!(
        ::BeliefPropagationProblem,
        ::AIE.NestedAlgorithm,
        state::AIE.State,
        substate::ITNNP.DaggerState
    )
    dst_cache = state.iterate.iterate

    state.iterate.maxdiff = 0.0

    for remote_result in substate.remote_results
        get_maxdiff = dtask -> dtask.iterate.maxdiff
        src_maxdiff = fetch(Dagger.@spawn get_maxdiff(remote_result))

        if src_maxdiff > state.iterate.maxdiff
            state.iterate.maxdiff = src_maxdiff
        end
    end

    function transfer_edges!(dst_chunk, src_chunk, edges)
        src_subcache = src_chunk.iterate
        dst_subcache = dst_chunk.iterate
        for edge in edges
            dst_subcache[edge] = src_subcache[edge]
        end
        return
    end

    transfer_dtasks = map(quotientedges(dst_cache)) do quotient_edge
        src_subcache = dst_cache[src(quotient_edge)]
        dst_subcache = dst_cache[dst(quotient_edge)]
        return Dagger.@spawn transfer_edges!(
            dst_subcache,
            fetch(src_subcache),
            edges(dst_cache, quotient_edge)
        )
    end

    wait.(transfer_dtasks)

    return state
end
