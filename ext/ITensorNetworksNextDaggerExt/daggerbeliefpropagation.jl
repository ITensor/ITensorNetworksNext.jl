import ITensorNetworksNext.ITensorNetworksNextParallel as ITNNP
using Dagger
using DataGraphs: DataGraphs, edge_data, get_edge_data, get_vertex_data, is_edge_assigned,
    is_vertex_assigned, set_edge_data!, set_vertex_data!, underlying_graph
using Dictionaries: Dictionary, Indices, getindices
using Graphs: AbstractEdge, AbstractGraph, dst, edges, src, vertices
using ITensorNetworksNext: ITensorNetworksNext, BeliefPropagation, BeliefPropagationCache,
    BeliefPropagationProblem, BeliefPropagationState, beliefpropagation,
    forest_cover_edge_sequence, select_algorithm, subcache
using NamedGraphs.GraphsExtensions: boundary_edges
using NamedGraphs.PartitionedGraphs: QuotientVertex, quotientedges, quotientvertices
using NamedGraphs: NamedGraphs

const DaggerBeliefPropagation = BeliefPropagation{<:ITNNP.DaggerNestedAlgorithm};

function ITNNP.DaggerBeliefPropagationCache(
        network::AbstractGraph;
        workers = nothing,
        scopes = nothing
    )
    underlying_cache = BeliefPropagationCache(network)

    keys = Indices(quotientvertices(underlying_cache))

    if isnothing(scopes)
        workers = isnothing(workers) ? Dagger.Distributed.workers() : workers

        sorted_workers = Iterators.take(Iterators.cycle(workers), length(keys))

        scopes = map(Dagger.ProcessScope, collect(sorted_workers))
    else
        if length(keys) != length(scopes)
            throw(
                ArgumentError(
                    "Number of provided scopes must match the number of vertex partitions of underlying graph"
                )
            )
        end
    end

    scope_dict = Dictionary(keys, scopes)

    quotient_chunks = map(keys) do quotient_vertex
        scope = scope_dict[quotient_vertex]
        iterate = subcache(underlying_cache, quotient_vertex)
        chunk = Dagger.@mutable scope = scope BeliefPropagationState(; iterate)
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
    return is_edge_assigned(bpc.underlying_cache, edge)
end

function DataGraphs.get_vertex_data(bpc::ITNNP.DaggerBeliefPropagationCache, vertex)
    return get_vertex_data(bpc.underlying_cache, vertex)
end
function DataGraphs.get_edge_data(
        bpc::ITNNP.DaggerBeliefPropagationCache,
        edge::AbstractEdge
    )
    return get_edge_data(bpc.underlying_cache, edge)
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
        kwargs...
    )
    return ITNNP.dagger_algorithm(quotientvertices(cache)) do quotient_vertex
        substate = fetch(cache[quotient_vertex])
        subcache = substate.iterate

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
        problem::BeliefPropagationProblem,
        algorithm::AIE.NestedAlgorithm,
        state::AIE.State,
        substate::ITNNP.DaggerState
    )
    dst_cache = state.iterate.iterate

    state.iterate.maxdiff = 0.0

    maxdiff_dtasks = map(substate.remote_results) do remote_result
        return Dagger.spawn(dtask -> dtask.iterate.maxdiff, remote_result)
    end

    maxdiff = maximum(fetch, maxdiff_dtasks)

    if maxdiff > state.iterate.maxdiff
        state.iterate.maxdiff = maxdiff
    end

    transfer_dtasks = map(quotientedges(dst_cache)) do quotient_edge
        src_subcache = dst_cache[src(quotient_edge)]
        dst_subcache = dst_cache[dst(quotient_edge)]

        src_subcache = fetch(src_subcache)

        return Dagger.spawn(
            dst_subcache,
            fetch(src_subcache),
            edges(dst_cache, quotient_edge)
        ) do dst, src, edges
            src_subcache = src.iterate
            dst_subcache = dst.iterate
            for edge in edges
                dst_subcache[edge] = src_subcache[edge]
            end
        end
    end

    foreach(wait, transfer_dtasks)

    return state
end

function ITNNP.finalize_state!(
        ::BeliefPropagationProblem,
        ::BeliefPropagation,
        state::ITNNP.DaggerState
    )
    dst_cache = state.iterate.iterate

    for quotient_vertex in quotientvertices(dst_cache)
        substate = fetch(dst_cache[quotient_vertex])
        subcache = substate.iterate
        edge_data(dst_cache) .= edge_data(subcache)
    end

    return state
end
