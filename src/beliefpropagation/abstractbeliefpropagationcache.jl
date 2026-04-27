using DataGraphs: AbstractDataGraph, edge_data, edge_data_type, vertex_data
using Graphs: AbstractEdge, AbstractGraph
using NamedGraphs.GraphsExtensions: boundary_edges
using NamedGraphs.PartitionedGraphs: QuotientEdge, QuotientView, parent
using NamedGraphs: AbstractEdges, AbstractVertices, to_graph_index

messages(bpc::AbstractDataGraph) = edge_data(bpc)
messages(bpc::AbstractGraph, edges) = map(e -> message(bpc, e), edges)

message(bpc::AbstractGraph, edge) = messages(bpc)[edge]

deletemessage!(bpc::AbstractGraph, edge) = not_implemented()

function deletemessages!(bpc::AbstractGraph, edges = edges(bpc))
    for e in edges
        deletemessage!(bpc, e)
    end
    return bpc
end

# Fallback; assume `setindex!` is implemented.
function setmessage!(bpc::AbstractGraph, edge, message)
    bpc[edge] = message
    return bpc
end
function setmessages!(bpc::AbstractGraph, messages)
    for (key, val) in messages
        setmessage!(bpc, key, val)
    end
    return bpc
end
function setmessages!(bpc_dst::AbstractGraph, bpc_src::AbstractGraph, edges)
    for e in edges
        setmessage!(bpc_dst, e, message(bpc_src, e))
    end
    return bpc_dst
end

factors(bpc::AbstractDataGraph) = vertex_data(bpc)
factors(bpc::AbstractGraph, vertices) = map(v -> factor(bpc, v), vertices)

factor(bpc::AbstractGraph, vertex) = bpc[vertex]

function setfactor!(bpc::AbstractGraph, vertex, factor)
    bpc[vertex] = factor
    return bpc
end

# Internal convenience only
_graph_index_scalar(bpc::AbstractGraph, vertex) = vertex_scalar(bpc, vertex)
_graph_index_scalar(bpc::AbstractGraph, edge::AbstractEdge) = edge_scalar(bpc, edge)

function edge_scalar(bp_cache::AbstractGraph, edge; kwargs...)
    m1s = messages(bp_cache, [edge])
    m2s = messages(bp_cache, [reverse(edge)])
    return contract_network(vcat(m1s, m2s); kwargs...)[]
end

function vertex_scalar(bp_cache::AbstractGraph, vertex; kwargs...)
    messages = incoming_messages(bp_cache, vertex)
    state = factors(bp_cache, [vertex])

    return contract_network(vcat(messages, state); kwargs...)[]
end

message_type(bpc::AbstractGraph) = message_type(typeof(bpc))
message_type(G::Type{<:AbstractGraph}) = eltype(Base.promote_op(messages, G))
message_type(type::Type{<:AbstractDataGraph}) = edge_data_type(type)

function vertex_scalars(bp_cache::AbstractGraph, vertices = vertices(bp_cache))
    return map(v -> vertex_scalar(bp_cache, v), vertices)
end

function edge_scalars(
        bp_cache::AbstractGraph,
        edges = edges(undirected_graph(underlying_graph(bp_cache)))
    )
    return map(e -> edge_scalar(bp_cache, e), edges)
end

function region_scalar(bpc::AbstractGraph, region)
    return mapreduce(ind -> _graph_index_scalar(bpc, ind), *, region)
end

function incoming_messages(bp_cache::AbstractGraph, vertices; ignore_edges = [])
    b_edges = boundary_edges(bp_cache, [vertices;]; dir = :in)
    b_edges = !isempty(ignore_edges) ? setdiff(b_edges, ignore_edges) : b_edges
    return messages(bp_cache, b_edges)
end

default_messages(::AbstractGraph) = not_implemented()

#Adapt interface for changing device
map_messages(f, bp_cache, es = edges(bp_cache)) = map_messages!(f, copy(bp_cache), es)
function map_messages!(f, bp_cache, es = edges(bp_cache))
    for e in es
        setmessage!(bp_cache, e, f(message(bp_cache, e)))
    end
    return bp_cache
end

map_factors(f, bp_cache, vs = vertices(bp_cache)) = map_factors!(f, copy(bp_cache), vs)
function map_factors!(f, bp_cache, vs = vertices(bp_cache))
    for v in vs
        setfactor!(bp_cache, v, f(factor(bp_cache, v)))
    end
    return bp_cache
end

adapt_messages(to, bp_cache, es = edges(bp_cache)) = map_messages(adapt(to), bp_cache, es)
adapt_factors(to, bp_cache, vs = vertices(bp_cache)) = map_factors(adapt(to), bp_cache, vs)

abstract type AbstractBeliefPropagationCache{V, VD, ED} <: AbstractDataGraph{V, VD, ED} end

factor_type(bpc::AbstractBeliefPropagationCache) = typeof(bpc)
factor_type(::Type{<:AbstractBeliefPropagationCache{<:Any, VD}}) where {VD} = VD

message_type(bpc::AbstractBeliefPropagationCache) = message_type(typeof(bpc))
message_type(::Type{<:AbstractBeliefPropagationCache{<:Any, <:Any, ED}}) where {ED} = ED

function logscalar(bpc::AbstractBeliefPropagationCache)
    numerator_terms = vertex_scalars(bpc)
    denominator_terms = edge_scalars(bpc)

    if any(t -> real(t) < 0, numerator_terms)
        numerator_terms = complex.(numerator_terms)
    end
    if any(t -> real(t) < 0, denominator_terms)
        denominator_terms = complex.(denominator_terms)
    end

    if any(iszero, denominator_terms)
        return -Inf
    end

    return sum(log.(numerator_terms)) - sum(log.((denominator_terms)))
end
scalar(bp_cache::AbstractBeliefPropagationCache) = exp(logscalar(bp_cache))
