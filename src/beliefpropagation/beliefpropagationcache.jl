using DataGraphs: AbstractDataGraph, DataGraphs, edge_data, edge_data_type,
    set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data,
    vertex_data_type
using Dictionaries: Dictionary, delete!, set!, getindices
using Graphs: AbstractGraph, connected_components, is_tree, is_directed
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy, parenttype
using NamedGraphs.GraphsExtensions: default_root_vertex, forest_cover, post_order_dfs_edges, undirected_graph, vertextype
using NamedGraphs.PartitionedGraphs: QuotientEdge, QuotientView, quotient_graph

using NamedGraphs: Vertices, convert_vertextype, parent_graph_indices

struct BeliefPropagationCache{V, VD, ED, E, G <: AbstractGraph{V}} <: AbstractBeliefPropagationCache{V, VD, ED}
    underlying_graph::G # we only use this for the edges.
    factors::Dictionary{V, VD}
    messages::Dictionary{E, ED}
    function BeliefPropagationCache(graph::AbstractGraph, factors::Dictionary, messages::Dictionary)
        # Ensure the graph is directed, if not make it directed.
        digraph = is_directed(graph) ? graph : directed_graph(graph)

        V = keytype(factors)
        VD = eltype(factors)

        E = keytype(messages)
        ED = eltype(messages)

        bpc = new{V, VD, ED, E, typeof(digraph)}(digraph, factors, messages)

        for edge in edges(bpc)
            get!(() -> default_message(bpc, edge), messages, edge)
        end
        return bpc
    end
end

DataGraphs.underlying_graph(bpc::BeliefPropagationCache) = bpc.underlying_graph

DataGraphs.is_vertex_assigned(bpc::BeliefPropagationCache, vertex) = haskey(bpc.factors, vertex)
DataGraphs.is_edge_assigned(bpc::BeliefPropagationCache, edge) = haskey(bpc.messages, edge)

DataGraphs.get_vertex_data(bpc::BeliefPropagationCache, vertex) = bpc.factors[vertex]
DataGraphs.get_edge_data(bpc::BeliefPropagationCache, edge::AbstractEdge) = bpc.messages[edge]

DataGraphs.set_vertex_data!(bpc::BeliefPropagationCache, val, vertex) = set!(bpc.factors, vertex, val)
DataGraphs.set_edge_data!(bpc::BeliefPropagationCache, val, edge) = set!(bpc.messages, edge, val)

# These two methods assume `network` behaves llike a tensor network
# (could be e.g. a QuotientView) otherwise how would one know what the factors should be.
function BeliefPropagationCache(network::AbstractGraph)
    graph = underlying_graph(network)
    return BeliefPropagationCache(graph, copy(vertex_data(network)))
end
function BeliefPropagationCache(MT::Type, network::AbstractGraph)
    graph = underlying_graph(network)
    return BeliefPropagationCache(MT, graph, copy(vertex_data(network)))
end

function BeliefPropagationCache(graph::AbstractGraph, factors::Dictionary)
    MT = vertex_data_type(typeof(graph))
    return BeliefPropagationCache(MT, graph, factors)
end
function BeliefPropagationCache(MT::Type, graph::AbstractGraph, factors::Dictionary)
    messages = Dictionary{edgetype(graph), MT}()
    return BeliefPropagationCache(graph, factors, messages)
end

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(bp_cache.underlying_graph), copy(bp_cache.factors), copy(bp_cache.messages))
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

function induced_subgraph_bpcache(graph, subvertices)
    underlying_subgraph, vlist = Graphs.induced_subgraph(underlying_graph(graph), subvertices)

    assigned = v -> isassigned(graph, v)

    assigned_subvertices = Iterators.filter(assigned, subvertices)
    assigned_subedges = Iterators.filter(assigned, edges(underlying_subgraph))

    factors = getindices(vertex_data(graph), Indices(assigned_subvertices))
    messages = getindices(edge_data(graph), Indices(assigned_subedges))

    subgraph = BeliefPropagationCache(underlying_subgraph, factors, messages)

    return subgraph, vlist
end

function NamedGraphs.induced_subgraph_from_vertices(graph::BeliefPropagationCache, subvertices)
    return induced_subgraph_bpcache(graph, subvertices)
end

## PartitionedGraphs

# Take a QuotientView of the underlying graph.
function PartitionedGraphs.quotientview(bpc::BeliefPropagationCache)

    graph = underlying_graph(bpc)

    quotient_view = QuotientView(graph)

    factors = map(v -> bpc[QuotientVertex(v)], Indices(vertices(quotient_view)))
    messages = map(e -> bpc[QuotientEdge(e)], Indices(edges(quotient_view)))

    return BeliefPropagationCache(quotient_view, factors, messages)
end

function default_message(bpc::BeliefPropagationCache, edge)
    return default_message(message_type(bpc), bpc[src(edge)], bpc[dst(edge)])
end
function default_message(T::Type, src, dst)
    array = ones(Tuple(inds(src) ∩ inds(dst)))
    return convert(T, array)
end
function default_message(T::Type{<:LazyNamedDimsArray}, src, dst)
    message = default_message(parenttype(T), src, dst)
    return convert(T, lazy(message))
end

NamedGraphs.to_graph_index(::BeliefPropagationCache, vertex::QuotientVertex) = vertex
# When getting data according the quotient vertices, take a lazy contraction.
function DataGraphs.get_index_data(tn::BeliefPropagationCache, vertex::QuotientVertex)
    data = collect(map(v -> tn[v], vertices(tn, vertex)))
    return mapreduce(lazy, *, data)
end
function DataGraphs.is_graph_index_assigned(tn::BeliefPropagationCache, vertex::QuotientVertex)
    return isassigned(tn, Vertices(vertices(tn, vertex)))
end
