using DataGraphs: DataGraphs, AbstractDataGraph, AbstractEdgeDataGraph, edge_data,
    edge_data_type, set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data,
    vertex_data_type
using Dictionaries: Dictionary, delete!, getindices, set!
using Graphs: AbstractGraph, connected_components, is_directed, is_tree
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy, parenttype
using NamedGraphs.GraphsExtensions: IsDirected, boundary_edges, default_root_vertex,
    directed_graph, forest_cover, in_incident_edges, post_order_dfs_edges, undirected_graph,
    vertextype
using NamedGraphs.PartitionedGraphs: QuotientEdge, QuotientView, quotient_graph
using NamedGraphs: AbstractNamedEdge, NamedDiGraph, NamedEdge, Vertices, convert_vertextype,
    ordered_vertices, parent_graph_indices, position_graph, to_graph_index, vertex_positions

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

Base.keys(cache::MessageCache) = edges(cache)

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
    copyto!(cache, messages)
    return cache
end

messagecache(pairs) = MessageCache(Dict(pairs))
messagecache(f, edges) = messagecache(edge => f(edge) for edge in edges)

function Graphs.rem_edge!(c::MessageCache, edge)
    delete!(c.messages, to_graph_index(c, edge))
    rem_edge!(c.underlying_graph, edge)
    return c
end

function Graphs.add_vertex!(c::MessageCache, vertex)
    add_edge!(c.underlying_graph, vertex)
    return c
end

function Graphs.has_edge(c::MessageCache, edge::AbstractNamedEdge)
    return has_edge(c.underlying_graph, edge)
end

# ================================ NamedGraphs interface ================================= #

function NamedGraphs.similar_graph(::Type{<:MessageCache}, vertices)
    return MessageCache(undef, vertices)
end

function NamedGraphs.similar_graph(::MessageCache, ED::Type, vertices::Vertices)
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
    tensors = vcat([factors[vertex]], collect(in_messages))
    return contract_network(tensors; kwargs...)[]
end

vertex_scalars(factors, messages) = vertex_scalars(factors, messages, keys(factors))
function vertex_scalars(factors::AbstractGraph, messages)
    return vertex_scalars(factors, messages, vertices(factors))
end
function vertex_scalars(factors, messages, vertices)
    return map(v -> vertex_scalar(factors, messages, v), vertices)
end

function edge_scalar(cache, edge; kwargs...)
    m1 = cache[edge]
    m2 = cache[reverse(edge)]
    return contract_network([m1, m2]; kwargs...)[]
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
