using DataGraphs: DataGraphs, AbstractDataGraph, edge_data, edge_data_type,
    set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data, vertex_data_type
using Dictionaries: Dictionary, delete!, getindices, set!
using Graphs: AbstractGraph, connected_components, is_directed, is_tree
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy, parenttype
using NamedGraphs.GraphsExtensions: IsDirected, boundary_edges, default_root_vertex,
    directed_graph, forest_cover, in_incident_edges, post_order_dfs_edges, undirected_graph,
    vertextype
using NamedGraphs.PartitionedGraphs: QuotientEdge, QuotientView, quotient_graph
using NamedGraphs: NamedDiGraph, Vertices, convert_vertextype, ordered_vertices,
    parent_graph_indices, position_graph, to_graph_index, vertex_positions

struct MessageCache{T, V} <: AbstractDataGraph{V, Nothing, T}
    messages::Dictionary{NamedEdge{V}, T}
    underlying_graph::NamedDiGraph{V}
    function MessageCache{T, V}(::UndefInitializer, vertices) where {T, V}
        messages = Dictionary{NamedEdge{V}, T}()
        underlying_graph = NamedDiGraph{V}(vertices)
        return new{T, V}(messages, underlying_graph)
    end
end

# single type parameter version of the inner constructor
function MessageCache{T}(::UndefInitializer, vertices) where {T}
    return MessageCache{T, eltype(vertices)}(undef, vertices)
end

# compatibility with generic key-val iterables
Base.keytype(c::MessageCache) = keytype(typeof(c))
Base.keytype(::Type{<:MessageCache{T, V}}) where {T, V} = NamedEdge{V}

Base.valtype(c::MessageCache) = valtype(typeof(c))
Base.valtype(::Type{<:MessageCache{T}}) where {T} = T

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
    add_edges!(cache.underlying_graph, edges)
    copyto!(cache, messages)
    return cache
end

messagecache(pairs) = MessageCache(Dict(pairs))

# ================================ NamedGraphs interface ================================= #
function NamedGraphs.add_edge!(c::MessageCache, edge)
    add_edge!(c.underlying_graph, edge)
    return c
end

function NamedGraphs.rem_edge!(c::MessageCache, edge)
    delete!(c.messages, to_graph_index(c, edge))
    rem_edge!(c.underlying_graph, edge)
    return c
end

# ================================= DataGraphs interface ================================= #

DataGraphs.underlying_graph(cache::MessageCache) = cache.underlying_graph

DataGraphs.is_vertex_assigned(::MessageCache, _) = false
DataGraphs.is_edge_assigned(c::MessageCache, edge) = haskey(c.messages, edge)

function DataGraphs.get_edge_data(c::MessageCache, edge::AbstractEdge)
    return c.messages[edge]
end
function DataGraphs.set_edge_data!(c::MessageCache, val, edge)
    return set!(c.messages, edge, val)
end

Base.copy(cache::MessageCache) = MessageCache(copy(cache.messages))

function Base.:(==)(cache1::MessageCache, cache2::MessageCache)
    ug1 = cache1.underlying_graph
    ug2 = cache2.underlying_graph

    ms1 = cache1.messages
    ms2 = cache2.messages

    return (ug1 == ug2 && ms1 == ms2)
end

function NamedGraphs.induced_subgraph_from_vertices(cache::MessageCache, subvertices)
    # TODO: once we have `subgraph_edges` in `NamedGraphs`, simplify this.
    underlying_subgraph, vlist =
        Graphs.induced_subgraph(cache.underlying_graph, subvertices)

    assigned = v -> isassigned(cache, v)

    assigned_subedges = Iterators.filter(assigned, edges(underlying_subgraph))

    messages = getindices(cache.messages, Indices(assigned_subedges))

    return MessageCache(messages), vlist
end

# see: copyto!(dest, src) for analogous behaviour to 2 argument method
# see: copyto!(dest, Rdest::CartesianIndices, src, Rsrc::CartesianIndices)
# for analogous behaviour to 3 argument method.
# TODO: these can be made generic for `AbtractDataGraph` in `DataGraphs.jl`
function copyto!_messagecache(
        cache_dst::MessageCache,
        cache_src,
        inds = nothing
    )
    inds = isnothing(inds) ? Indices(keys(cache_src)) : Indices(inds)
    view(edge_data(cache_dst), inds) .= view(cache_src, inds)
    return cache_dst
end

function Base.copyto!(
        cache_dst::MessageCache,
        cache_src::AbstractDataGraph,
        inds = nothing
    )
    copyto!_messagecache(cache_dst, edge_data(cache_src), inds)
    return cache_dst
end

function Base.copyto!(
        cache_dst::MessageCache,
        dictionary_src::Dictionary,
        inds = nothing
    )
    copyto!_messagecache(cache_dst, dictionary_src, inds)
    return cache_dst
end

function Base.copyto!(
        cache_dst::MessageCache,
        dict_src::Dict,
        inds = keys(dict_src)
    )
    for key in inds
        cache_dst[key] = dict_src[key]
    end
    return cache_dst
end

# ===================================== contraction ====================================== #

function incoming_messages(cache::AbstractGraph, pair::Pair)
    edge = to_graph_index(cache, pair)
    return incoming_messages(cache, edge)
end
function incoming_messages(cache::AbstractGraph, edge::AbstractEdge)
    inds = Indices(in_incident_edges(cache, src(edge)))
    return getindices(cache, filter(e -> e != reverse(edge), inds))
end

function environment_messages(cache::AbstractGraph, vertices)
    inds = Indices(boundary_edges(cache, vertices; dir = :in))
    return getindices(cache, inds)
end

function vertex_scalar(factors, messages, vertex; kwargs...)
    in_messages = environment_messages(messages, [vertex])
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
function logscalar(factors, messages)
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

scalar(factors, messages) = exp(logscalar(factors, messages))

# TODO: This needs to go in NamedGraphs.GraphsExtensions
function forest_cover_edge_sequence(gi::AbstractGraph; root_vertex = default_root_vertex)
    # All we care about are the edges so the type of the graph doesnt matter
    g = similar_graph(NamedGraph, vertices(gi))
    add_edges!(g, edges(gi))
    forests = forest_cover(g)
    rv = edgetype(g)[]
    for forest in forests
        trees = [forest[Vertices(vs)] for vs in connected_components(forest)]
        for tree in trees
            tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
            push!(rv, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
        end
    end
    return rv
end

# ======================================= printing ======================================= #

# TODO: This is the definition for the proposed `DataGraphs.AbstractEdgeDataGraph`.
function Base.show(io::IO, mime::MIME"text/plain", graph::MessageCache)
    println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
    show(io, mime, vertices(graph))
    println(io, "\n")
    println(io, "and $(ne(graph)) edge(s):")
    for e in edges(graph)
        show(io, mime, e)
        println(io)
    end
    println(io)
    println(io, "with edge data:")
    show(io, mime, edge_data(graph))
    return nothing
end

Base.show(io::IO, graph::MessageCache) = show(io, MIME"text/plain"(), graph)
