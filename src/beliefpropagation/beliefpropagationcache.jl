using DataGraphs:
    DataGraphs,
    AbstractDataGraph,
    DataGraph,
    get_vertex_data,
    get_edge_data,
    set_vertex_data!,
    set_edge_data!,
    vertex_data_type,
    edge_data_type,
    underlying_graph,
    underlying_graph_type,
    is_vertex_assigned,
    is_edge_assigned
using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components, is_directed
using NamedGraphs: NamedDiGraph, convert_vertextype, parent_graph_indices, Vertices
using NamedGraphs.GraphsExtensions: default_root_vertex,
    forest_cover,
    post_order_dfs_edges,
    vertextype,
    is_path_graph,
    undirected_graph
using NamedGraphs.PartitionedGraphs: QuotientView, QuotientEdge, QuotientEdges, quotient_graph, quotientedges
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy, parenttype

struct BeliefPropagationCache{V, VD, ED, G <: AbstractGraph{V}, N <: AbstractGraph{V}, E} <:
    AbstractBeliefPropagationCache{V, VD, ED}
    underlying_graph::G # we only use this for the edges.
    network::N
    messages::Dictionary{E, ED}
    function BeliefPropagationCache(network::AbstractGraph, messages::Dictionary)

        V = vertextype(network)
        VD = vertex_data_type(network)
        N = typeof(network)
        ET = keytype(messages)
        ED = eltype(messages)

        # Construct a directed graph version of the underlying graph of the tensor network.
        digraph = directed_graph(underlying_graph(network))

        bpc = new{V, VD, ED, typeof(digraph), N, ET}(digraph, network, messages)

        for edge in edges(bpc)
            get!(() -> default_message(bpc, edge), messages, edge)
        end
        return bpc
    end
end

network(bp_cache) = getfield(bp_cache, :network)

DataGraphs.underlying_graph(bpc::BeliefPropagationCache) = getfield(bpc, :underlying_graph)

DataGraphs.is_vertex_assigned(bpc::BeliefPropagationCache, vertex) = is_vertex_assigned(network(bpc), vertex)
DataGraphs.is_edge_assigned(bpc::BeliefPropagationCache, edge) = haskey(bpc.messages, edge)

DataGraphs.get_vertex_data(bpc::BeliefPropagationCache, vertex) = get_vertex_data(network(bpc), vertex)
DataGraphs.get_edge_data(bpc::BeliefPropagationCache, edge::AbstractEdge) = bpc.messages[edge]

DataGraphs.set_vertex_data!(bpc::BeliefPropagationCache, val, vertex) = set_vertex_data!(network(bpc), val, vertex)
DataGraphs.set_edge_data!(bpc::BeliefPropagationCache, val, edge) = set!(bpc.messages, edge, val)

function BeliefPropagationCache(network::AbstractGraph)
    MT = vertex_data_type(typeof(network))
    return BeliefPropagationCache(MT, network)
end
function BeliefPropagationCache(MT::Type, network::AbstractGraph)
    dict = Dictionary{edgetype(network), MT}()
    return BeliefPropagationCache(network, dict)
end

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(network(bp_cache)), copy(messages(bp_cache)))
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

function bpcache_induced_subgraph(graph, subvertices)
    underlying_subgraph, vlist = Graphs.induced_subgraph(network(graph), subvertices)

    edge_data = Dictionary{edgetype(underlying_subgraph), edge_data_type(typeof(graph))}()

    subgraph = BeliefPropagationCache(underlying_subgraph, edge_data)
    for e in edges(subgraph)
        if isassigned(graph, e)
            subgraph[e] = graph[e]
        end
    end
    return subgraph, vlist
end

function NamedGraphs.induced_subgraph_from_vertices(graph::BeliefPropagationCache, subvertices)
    return bpcache_induced_subgraph(graph, subvertices)
end

## PartitionedGraphs

function PartitionedGraphs.quotientview(bpc::BeliefPropagationCache)
    inds = Indices(parent_graph_indices(QuotientEdges(underlying_graph(bpc))))
    data = map(e -> bpc[QuotientEdge(e)], inds)
    return BeliefPropagationCache(QuotientView(network(bpc)), data)
end

function default_message(bpc::BeliefPropagationCache, edge)
    return default_message(message_type(bpc), network(bpc), edge)
end
function default_message(T::Type, network, edge)
    array = ones(Tuple(linkinds(network, edge)))
    return convert(T, array)
end
function default_message(T::Type{<:LazyNamedDimsArray}, network, edge)
    message = default_message(parenttype(T), network, edge)
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
