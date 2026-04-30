using DataGraphs: DataGraphs, AbstractDataGraph, edge_data, edge_data_type,
    set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data, vertex_data_type
using Dictionaries: Dictionary, delete!, getindices, set!
using Graphs: AbstractGraph, connected_components, is_directed, is_tree
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy, parenttype
using NamedGraphs.GraphsExtensions: IsDirected, default_root_vertex, directed_graph,
    forest_cover, post_order_dfs_edges, undirected_graph, vertextype
using NamedGraphs.PartitionedGraphs: QuotientEdge, QuotientView, quotient_graph
using NamedGraphs:
    NamedDiGraph, Vertices, convert_vertextype, parent_graph_indices, to_graph_index

struct MessageCache{MT, V, E} <: AbstractDataGraph{V, Nothing, MT}
    messages::Dictionary{E, MT}
    underlying_graph::NamedDiGraph{V}
    global function _MessageCache(
            messages::Dictionary{E, MT},
            underlying_graph::NamedDiGraph{V}
        ) where {MT, V, E}
        return new{MT, V, E}(messages, underlying_graph)
    end
end

DataGraphs.underlying_graph(c::MessageCache) = c.underlying_graph

DataGraphs.is_vertex_assigned(::MessageCache, _) = false
DataGraphs.is_edge_assigned(c::MessageCache, edge) = haskey(c.messages, edge)

function DataGraphs.get_edge_data(c::MessageCache, edge::AbstractEdge)
    return c.messages[edge]
end
function DataGraphs.set_edge_data!(c::MessageCache, val, edge)
    return set!(c.messages, edge, val)
end

# Utility function for constructing a directed graph with existing edges + all reverses.
function _message_cache_underlying_graph(graph::AbstractGraph)
    digraph = similar_graph(NamedDiGraph, vertices(graph))
    for edge in edges(graph)
        add_edge!(digraph, edge)
        if !is_directed(graph)
            add_edge!(digraph, reverse(edge))
        end
    end
    return digraph
end

MessageCache(::UndefInitializer, graph::AbstractGraph) = MessageCache{Any}(undef, graph)

function MessageCache{ED}(::UndefInitializer, graph::AbstractGraph) where {ED}
    messages = Dictionary{edgetype(graph), ED}()
    return MessageCache(messages, graph)
end

function MessageCache(f::Function, graph::AbstractGraph)
    digraph = _message_cache_underlying_graph(graph)
    messages = map(f, Indices(edges(digraph)))
    return MessageCache(messages, digraph)
end

function MessageCache(messages, graph::AbstractGraph)
    digraph = _message_cache_underlying_graph(graph)
    return _MessageCache(Dictionary(messages), digraph) # Call the inner constructor.
end

function Base.copy(cache::MessageCache)
    return MessageCache(copy(cache.messages), copy(cache.underlying_graph))
end

function Base.:(==)(cache1::MessageCache, cache2::MessageCache)
    if cache1.underlying_graph != cache2.underlying_graph
        return false
    elseif cache1.messages != cache2.messages
        return false
    end
    return true
end

function NamedGraphs.induced_subgraph_from_vertices(cache::MessageCache, subvertices)
    underlying_subgraph, vlist =
        Graphs.induced_subgraph(cache.underlying_graph, subvertices)

    assigned = v -> isassigned(cache, v)

    assigned_subedges = Iterators.filter(assigned, edges(underlying_subgraph))

    messages = getindices(edge_data(cache), Indices(assigned_subedges))

    subgraph = MessageCache(messages, underlying_subgraph)

    return subgraph, vlist
end

# TODO: This needs to go in GraphsExtensions
function forest_cover_edge_sequence(gi::AbstractGraph; root_vertex = default_root_vertex)
    # All we care about are the edges so the type of the graph doesnt matter
    g = NamedGraph(vertices(gi))
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

# =============================== message/factor interface =============================== #

message_type(::Type) = not_implemented()
message_type(cache) = message_type(typeof(cache))
message_type(T::Type{<:MessageCache}) = edge_data_type(T)

factor(_factors, _vertex) = not_implemented()
factor(factors::AbstractGraph, vertex) = factors[vertex]

function factors(all_factors, vertices)
    return map(vertex -> factor(all_factors, vertex), vertices)
end

# Specific for graphs
function factors(all_factors::AbstractGraph)
    return map(vertex -> factor(all_factors, vertex), vertices(all_factors))
end

message(_messages, _edge) = not_implemented()
message(messages::AbstractGraph, edge) = messages[edge]

function messages(all_messages, edges)
    return map(edge -> message(all_messages, edge), edges)
end

# Specific for graphs
function messages(all_messages::AbstractGraph)
    return map(edge -> message(all_messages, edge), edges(all_messages))
end

# Specific to the concrete type.
messages(cache::MessageCache) = cache.messages

function incoming_messages(cache::AbstractGraph, vertices; ignore_edges = [])
    b_edges = boundary_edges(cache, [vertices;]; dir = :in)
    if !isempty(ignore_edges)
        b_edges = setdiff(b_edges, to_graph_index(cache, ignore_edges))
    end
    return messages(cache, b_edges)
end

function setmessage!(cache::AbstractGraph, edge, message)
    cache[edge] = message
    return cache
end
function setmessages!(cache::AbstractGraph, messages)
    for (key, val) in messages
        setmessage!(cache, key, val)
    end
    return cache
end
function setmessages!(cache_dst::AbstractGraph, cache_src::AbstractGraph, edges)
    for e in edges
        setmessage!(cache_dst, e, message(cache_src, e))
    end
    return cache_dst
end

# =================================== adapt interface ==================================== #

map_messages(f, cache, es = edges(cache)) = map_messages!(f, copy(cache), es)
function map_messages!(f, cache, es = edges(cache))
    for e in es
        setmessage!(cache, e, f(message(cache, e)))
    end
    return cache
end

adapt_messages(to, cache, es = edges(cache)) = map_messages(adapt(to), cache, es)

# ===================================== contraction ====================================== #

function vertex_scalar(factors, messages, vertex; kwargs...)
    in_messages = incoming_messages(messages, vertex)
    state = [factor(factors, vertex)]
    return contract_network(vcat(in_messages, state); kwargs...)[]
end

vertex_scalars(factors, messages) = vertex_scalars(factors, messages, keys(factors))
function vertex_scalars(factors::AbstractGraph, messages)
    return vertex_scalars(factors, messages, vertices(factors))
end
function vertex_scalars(factors, messages, vertices)
    return map(v -> vertex_scalar(factors, messages, v), vertices)
end

function edge_scalar(cache, edge; kwargs...)
    m1s = messages(cache, [edge])
    m2s = messages(cache, [reverse(edge)])
    return contract_network(vcat(m1s, m2s); kwargs...)[]
end

edge_scalars(cache) = edge_scalars(cache, keys(cache))
edge_scalars(cache::AbstractGraph) = edge_scalars(cache, edges(cache))

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
