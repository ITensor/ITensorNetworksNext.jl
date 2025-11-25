using Graphs: AbstractGraph, AbstractEdge
using DataGraphs: AbstractDataGraph, edge_data, vertex_data, edge_data_eltype
using NamedGraphs.GraphsExtensions: boundary_edges
using NamedGraphs.PartitionedGraphs: QuotientView, QuotientEdge, parent

messages(::AbstractGraph) = not_implemented()
messages(bp_cache::AbstractDataGraph) = edge_data(bp_cache)
messages(bp_cache::AbstractGraph, edges) = [message(bp_cache, e) for e in edges]

message(bp_cache::AbstractGraph, edge::AbstractEdge) = messages(bp_cache)[edge]

deletemessage!(bp_cache::AbstractGraph, edge) = not_implemented()
function deletemessage!(bp_cache::AbstractDataGraph, edge)
    ms = messages(bp_cache)
    delete!(ms, edge)
    return bp_cache
end

function deletemessages!(bp_cache::AbstractGraph, edges = edges(bp_cache))
    for e in edges
        deletemessage!(bp_cache, e)
    end
    return bp_cache
end

setmessage!(bp_cache::AbstractGraph, edge, message) = not_implemented()
function setmessage!(bp_cache::AbstractDataGraph, edge, message)
    ms = messages(bp_cache)
    set!(ms, edge, message)
    return bp_cache
end
function setmessage!(bp_cache::QuotientView, edge, message)
    setmessages!(parent(bp_cache), QuotientEdge(edge), message)
    return bp_cache
end

function setmessages!(bp_cache::AbstractGraph, edge::QuotientEdge, message)
    for e in edges(bp_cache, edge)
        setmessage!(parent(bp_cache), e, message[e])
    end
    return bp_cache
end
function setmessages!(bpc_dst::AbstractGraph, bpc_src::AbstractGraph, edges)
    for e in edges
        setmessage!(bpc_dst, e, message(bpc_src, e))
    end
    return bpc_dst
end

factors(bpc::AbstractGraph) = vertex_data(bpc)
factors(bpc::AbstractGraph, vertices::Vector) = [factor(bpc, v) for v in vertices]
factors(bpc::AbstractGraph{V}, vertex::V) where {V} = factors(bpc, V[vertex])

factor(bpc::AbstractGraph, vertex) = factors(bpc)[vertex]

setfactor!(bpc::AbstractGraph, vertex, factor) = not_implemented()
function setfactor!(bpc::AbstractDataGraph, vertex, factor)
    fs = factors(bpc)
    set!(fs, vertex, factor)
    return bpc
end

function region_scalar(bp_cache::AbstractGraph, edge::AbstractEdge)
    return message(bp_cache, edge) * message(bp_cache, reverse(edge))
end

function region_scalar(bp_cache::AbstractGraph, vertex)

    messages = incoming_messages(bp_cache, vertex)
    state = factors(bp_cache, vertex)

    return reduce(*, messages) * reduce(*, state)
end

message_type(bpc::AbstractGraph) = message_type(typeof(bpc))
message_type(G::Type{<:AbstractGraph}) = eltype(Base.promote_op(messages, G))
message_type(type::Type{<:AbstractDataGraph}) = edge_data_eltype(type)

function vertex_scalars(bp_cache::AbstractGraph, vertices = vertices(bp_cache))
    return map(v -> region_scalar(bp_cache, v), vertices)
end

function edge_scalars(bp_cache::AbstractGraph, edges = edges(bp_cache))
    return map(e -> region_scalar(bp_cache, e), edges)
end

function scalar_factors_quotient(bp_cache::AbstractGraph)
    return vertex_scalars(bp_cache), edge_scalars(bp_cache)
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

abstract type AbstractBeliefPropagationCache{V, ED} <: AbstractDataGraph{V, Nothing, ED} end

function free_energy(bp_cache::AbstractBeliefPropagationCache)
    numerator_terms, denominator_terms = scalar_factors_quotient(bp_cache)
    if any(t -> real(t) < 0, numerator_terms)
        numerator_terms = complex.(numerator_terms)
    end
    if any(t -> real(t) < 0, denominator_terms)
        denominator_terms = complex.(denominator_terms)
    end

    any(iszero, denominator_terms) && return -Inf
    return sum(log.(numerator_terms)) - sum(log.((denominator_terms)))
end
partitionfunction(bp_cache::AbstractBeliefPropagationCache) = exp(free_energy(bp_cache))
