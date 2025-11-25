using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph
using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs: convert_vertextype
using NamedGraphs.GraphsExtensions: default_root_vertex, forest_cover, post_order_dfs_edges, is_path_graph
using NamedGraphs.PartitionedGraphs: QuotientView, QuotientEdge, quotient_graph

struct BeliefPropagationCache{V, N <: AbstractGraph{V}, ET, MT} <:
    AbstractBeliefPropagationCache{V, MT}
    network::N
    messages::Dictionary{ET, MT}
end

network(bp_cache) = underlying_graph(bp_cache)

DataGraphs.underlying_graph(bpc::BeliefPropagationCache) = getfield(bpc, :network)
DataGraphs.edge_data(bpc::BeliefPropagationCache) = getfield(bpc, :messages)
DataGraphs.vertex_data(bpc::BeliefPropagationCache) = vertex_data(network(bpc))
function DataGraphs.underlying_graph_type(type::Type{<:BeliefPropagationCache})
    return fieldtype(type, :network)
end

message_type(::Type{<:BeliefPropagationCache{V, N, ET, MT}}) where {V, N, ET, MT} = MT

function BeliefPropagationCache(alg, network::AbstractGraph)
    es = collect(edges(network))
    es = vcat(es, reverse.(es))
    messages = map(edge -> default_message(alg, network, edge), es)
    return BeliefPropagationCache(network, Dictionary(es, messages))
end

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(network(bp_cache)), copy(messages(bp_cache)))
end

# TODO: This needs to go in DataGraphsGraphsExtensionsExt
#
# This function is problematic when `ng isa TensorNetwork` as it relies on deleting edges
# and taking subgraphs, which is not always well-defined for the `TensorNetwork` type,
# hence we just strip off any `AbstractDataGraph` data to avoid this.
function forest_cover_edge_sequence(g::AbstractDataGraph; kwargs...)
    return forest_cover_edge_sequence(underlying_graph(g); kwargs...)
end
# TODO: This needs to go in PartitionedGraphsGraphsExtensionsExt
#
# While it is not at all necessary to explictly instantiate the `QuotientView`, it allows the
# data of a data graph to be removed using the above method if `parent_type(g)` is an
# `AbstractDataGraph`.
function forest_cover_edge_sequence(g::QuotientView; kwargs...)
    return forest_cover_edge_sequence(quotient_graph(parent(g)); kwargs...)
end
# TODO: This needs to go in GraphsExtensions
function forest_cover_edge_sequence(g::AbstractGraph; root_vertex = default_root_vertex)
    add_edges!(g, edges(g))
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

function Graphs.induced_subgraph(graph::BeliefPropagationCache, subvertices)
    return bpcache_induced_subgraph(graph, subvertices)
end
# For method ambiguity
function Graphs.induced_subgraph(graph::BeliefPropagationCache{V}, subvertices::AbstractVector{V}) where {V <: Int}
    return bpcache_induced_subgraph(graph, subvertices)
end

## PartitionedGraphs

function PartitionedGraphs.quotientview(bpc::BeliefPropagationCache)
    qview = QuotientView(network(bpc))
    messages = edge_data(QuotientView(bpc))
    return BeliefPropagationCache(qview, messages)
end
