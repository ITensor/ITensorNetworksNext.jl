using DataGraphs:
    DataGraphs,
    AbstractDataGraph,
    DataGraph,
    has_edge_data,
    get_vertex_data,
    get_edge_data,
    set_vertex_data!,
    set_edge_data!,
    unset_vertex_data!,
    unset_edge_data!,
    vertex_data_eltype,
    edge_data_eltype,
    underlying_graph,
    underlying_graph_type
using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components, is_directed
using NamedGraphs: NamedDiGraph, convert_vertextype, parent_graph_indices
using NamedGraphs.GraphsExtensions: default_root_vertex,
    forest_cover,
    post_order_dfs_edges,
    vertextype,
    is_path_graph,
    undirected_graph
using NamedGraphs.PartitionedGraphs: QuotientView, QuotientEdge, QuotientEdges, quotient_graph, quotientedges

struct BeliefPropagationCache{V, G <: AbstractGraph{V}, N <: AbstractGraph{V}, ET, MT} <:
    AbstractBeliefPropagationCache{V, MT}
    underlying_graph::G # we only use this for the edges.
    network::N
    messages::Dictionary{ET, MT}
    function BeliefPropagationCache(network::AbstractGraph, messages::Dictionary)

        V = vertextype(network)
        N = typeof(network)
        ET = keytype(messages)
        MT = eltype(messages)

        # Construct a directed graph version of the underlying graph of the tensor network.
        digraph = directed_graph(underlying_graph(network))

        bpc = new{V, typeof(digraph), N, ET, MT}(digraph, network, messages)

        for edge in edges(bpc)
            get!(() -> default_message(bpc, edge), messages, edge)
        end
        return bpc
    end
end

network(bp_cache) = getfield(bp_cache, :network)

DataGraphs.underlying_graph(bpc::BeliefPropagationCache) = getfield(bpc, :underlying_graph)

DataGraphs.has_vertex_data(bpc::BeliefPropagationCache, vertex) = has_vertex_data(network(bpc), vertex)
DataGraphs.has_edge_data(bpc::BeliefPropagationCache, edge) = haskey(bpc.messages, edge)

DataGraphs.get_vertex_data(bpc::BeliefPropagationCache, vertex) = get_vertex_data(network(bpc), vertex)
DataGraphs.get_edge_data(bpc::BeliefPropagationCache, edge::AbstractEdge) = bpc.messages[edge]

DataGraphs.set_vertex_data!(bpc::BeliefPropagationCache, val, vertex) = set_vertex_data!(network(bpc), val, vertex)
DataGraphs.set_edge_data!(bpc::BeliefPropagationCache, val, edge) = set!(bpc.messages, edge, val)

DataGraphs.unset_vertex_data!(bpc::BeliefPropagationCache, val, vertex) = unset_vertex_data!(network(bpc), val, vertex)
DataGraphs.unset_edge_data!(bpc::BeliefPropagationCache, val, edge) = unset!(bpc.messages, edge, val)

function DataGraphs.vertex_data_eltype(T::Type{<:BeliefPropagationCache})
    return vertex_data_eltype(fieldtype(T, :network))
end
function DataGraphs.edge_data_eltype(T::Type{<:BeliefPropagationCache})
    return eltype(fieldtype(T, :messages))
end

message_type(T::Type{<:BeliefPropagationCache}) = edge_data_eltype(T)

function BeliefPropagationCache(network::AbstractGraph)
    MT = vertex_data_eltype(typeof(network))
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
        trees = [forest[vs] for vs in connected_components(forest)]
        for tree in trees
            tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
            push!(rv, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
        end
    end
    return rv
end

function bpcache_induced_subgraph(graph, subvertices)
    underlying_subgraph, vlist = Graphs.induced_subgraph(network(graph), subvertices)
    subgraph = BeliefPropagationCache(underlying_subgraph, typeof(edge_data(graph))())
    for e in edges(subgraph)
        if isassigned(graph, e)
            set!(edge_data(subgraph), e, graph[e])
        end
    end
    return subgraph, vlist
end

function Graphs.induced_subgraph(graph::BeliefPropagationCache{V}, subvertices::Vector{V}) where {V}
    return bpcache_induced_subgraph(graph, subvertices)
end

## PartitionedGraphs

function PartitionedGraphs.quotientview(bpc::BeliefPropagationCache)
    inds = Indices(parent_graph_indices(QuotientEdges(underlying_graph(bpc))))
    data = map(e -> bpc[QuotientEdge(e)], inds)
    return BeliefPropagationCache(QuotientView(network(bpc)), data)
end
