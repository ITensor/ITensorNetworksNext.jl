using DataGraphs: DataGraphs, AbstractDataGraph, AbstractEdgeDataGraph, edge_data,
    edge_data_type, set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data,
    vertex_data_type
using Dictionaries: Dictionary, getindices, set!, unset!
using Graphs: AbstractGraph, connected_components, is_directed, is_tree
using ITensorBase: state, unnamed
using NamedGraphs: AbstractNamedEdge, NamedDiGraph, NamedEdge, add_edges!, boundary_edges,
    in_incident_edges, to_graph_index, vertextype
using SplitApplyCombine: mapmany

struct MessageCache{T, V} <: AbstractEdgeDataGraph{T, V}
    messages::Dictionary{NamedEdge{V}, T}
    underlying_graph::NamedDiGraph{V}
    function MessageCache{T, V}(::UndefInitializer, vertices) where {T, V}
        messages = Dictionary{NamedEdge{V}, T}()
        underlying_graph = NamedDiGraph{V}(vertices)
        return new{T, V}(messages, underlying_graph)
    end
end

# single type parameter version of the inner constructor
function MessageCache(::UndefInitializer, vertices)
    return MessageCache{Any}(undef, vertices)
end
function MessageCache{T}(::UndefInitializer, vertices) where {T}
    return MessageCache{T, eltype(vertices)}(undef, vertices)
end

MessageCache(messages) = MessageCache{valtype(messages)}(messages)

function MessageCache{T}(messages) where {T}
    V = vertextype(keytype(messages))
    return MessageCache{T, V}(messages)
end

# `messages` is any iterable data structure, where `keys(messages)` are edges
# and the values are the messages on those edges.
function MessageCache{T, V}(messages) where {T, V}
    edges = keys(messages)
    vertices = union(src.(edges), dst.(edges))
    cache = MessageCache{T, V}(undef, vertices)
    add_edges!(cache.underlying_graph, edges)
    copyto!(cache, messages)
    return cache
end

messagecache(pairs) = MessageCache(Dict(pairs))
messagecache(f, edges) = messagecache(edge => f(edge) for edge in edges)

function Graphs.rem_edge!(c::MessageCache, edge)
    unset!(c.messages, to_graph_index(c, edge))
    return rem_edge!(c.underlying_graph, edge)
end

function Graphs.add_vertex!(c::MessageCache, vertex)
    return add_vertex!(c.underlying_graph, vertex)
end

function Graphs.has_edge(c::MessageCache, edge::AbstractNamedEdge)
    return has_edge(c.underlying_graph, edge)
end

# ================================ NamedGraphs interface ================================= #

function NamedGraphs.similar_graph(::Type{<:MessageCache}, vertices)
    return MessageCache(undef, vertices)
end

function NamedGraphs.similar_graph(cache::MessageCache, T::Type)
    new_cache = similar_graph(cache, T, vertices(cache))
    add_edges!(new_cache.underlying_graph, edges(cache))
    return new_cache
end
function NamedGraphs.similar_graph(::MessageCache, ED::Type, vertices)
    return MessageCache{ED}(undef, collect(vertices))
end

# ================================= DataGraphs interface ================================= #

DataGraphs.underlying_graph(cache::MessageCache) = cache.underlying_graph

DataGraphs.is_vertex_assigned(::MessageCache, _) = false
DataGraphs.is_edge_assigned(c::MessageCache, edge) = haskey(c.messages, edge)

DataGraphs.get_edge_data(c::MessageCache, edge::AbstractEdge) = c.messages[edge]
function DataGraphs.set_edge_data!(c::MessageCache, val, edge)
    has_edge(c, edge) || add_edge!(c.underlying_graph, edge)
    set!(c.messages, edge, val)
    return c
end

function DataGraphs.insert_edge_data!(cache::MessageCache, edge, val)
    add_edge!(cache.underlying_graph, edge)
    insert!(cache.messages, edge, val)
    return cache
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::MessageCache) = true
Dictionaries.isinsertable(::MessageCache) = true

function Base.map(f, cache::MessageCache)
    new_cache = similar_graph(cache, Base.promote_op(f, valtype(cache)))
    map!(f, new_cache, cache)
    return new_cache
end

function Base.map!(f, dst::MessageCache, src)
    for key in keys(src)
        dst[key] = f(src[key])
    end
    return dst
end

# ===================================== contraction ====================================== #

function incoming_messages(cache::AbstractGraph, pair::Pair)
    edge = to_graph_index(cache, pair)
    return incoming_messages(cache, edge)
end
function incoming_messages(cache::AbstractGraph, edge::AbstractEdge)
    dimnames = Indices(in_incident_edges(cache, src(edge)))
    return getindices(cache, filter(e -> e != reverse(edge), dimnames))
end

# TODO: maybe this should be defined in `DataGraphs`.
function incoming_edge_data(cache::AbstractGraph, vertices)
    dimnames = Indices(boundary_edges(cache, vertices; dir = :in))
    return getindices(cache, dimnames)
end

function vertex_scalar(factors, messages, vertex; kwargs...)
    in_messages = incoming_edge_data(messages, [vertex])
    tensors = [[factors[vertex]]; collect(in_messages)]
    return contract_network(tensors; kwargs...)[]
end

vertex_scalars(factors, messages) = vertex_scalars(factors, messages, keys(factors))
function vertex_scalars(factors::AbstractGraph, messages)
    return vertex_scalars(factors, messages, vertices(factors))
end
function vertex_scalars(factors, messages, vertices)
    return map(v -> vertex_scalar(factors, messages, v), vertices)
end

function edge_scalar(cache, edge)
    return (cache[edge] * cache[reverse(edge)])[]
end

edge_scalars(cache) = edge_scalars(cache, keys(cache))

function edge_scalars(cache, edges)
    processed = Set{eltype(edges)}()

    T = Base.promote_op(edge_scalar, typeof(cache), eltype(edges))

    scalars = T[]

    # Ignore repeated edges and their reverses.
    for e in edges
        if e in processed || reverse(e) in processed
            continue
        end
        push!(processed, e)
        push!(scalars, edge_scalar(cache, e))
    end

    return scalars
end

function region_scalar(factors, messages, region)
    return mapreduce(vertex -> vertex_scalar(factors, messages, vertex), *, region)
end

# We need a graph structure here, so assume `factors` is a graph.
function bethe_free_energy(factors, messages)
    numerator_terms = vertex_scalars(factors, messages)
    denominator_terms = edge_scalars(messages)

    if any(t -> real(t) < 0, numerator_terms)
        numerator_terms = complex.(numerator_terms)
    end
    if any(t -> real(t) < 0, denominator_terms)
        denominator_terms = complex.(denominator_terms)
    end

    if any(iszero, denominator_terms)
        return -Inf
    end

    return sum(log.(numerator_terms)) - sum(log.(denominator_terms))
end

# ===================================== NormNetwork ====================================== #

function similar_message_environment(nn::NormNetwork)
    messages = mapmany(vertices(nn)) do vertex
        return map(in_incident_edges(nn, vertex)) do edge
            braview = BraView(nn)
            ketview = KetView(nn)

            ketnames = linknames(ketview, edge)
            ketaxis = unnamed.(linkaxes(ketview, edge))

            branames = linknames(braview, edge)

            # Bond leg (ket) = operator output, bra-layer leg = input. Built on the src-side ket
            # axis, whose arrow is opposite the dst endpoint's bond, so the gauge contracts back
            # into the destination state.
            message = similar_operator(ketview[vertex], ketaxis, ketnames, branames)

            return edge => message
        end
    end

    return messagecache(messages)
end

function message_environment(f::Base.Callable, nn::NormNetwork)
    return map(f, similar_message_environment(nn))
end
