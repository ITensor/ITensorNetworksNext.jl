using Combinatorics: combinations
using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph
using Dictionaries: AbstractDictionary, Indices, dictionary, set!, unset!
using Graphs: AbstractSimpleGraph, rem_vertex!, rem_edge!
using NamedDimsArrays: AbstractNamedDimsArray, dimnames
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype, Vertices, parent_graph_indices
using NamedGraphs.GraphsExtensions: GraphsExtensions, arranged_edges, arrange_edge, vertextype
using NamedGraphs.PartitionedGraphs:
    AbstractPartitionedGraph,
    PartitionedGraphs,
    departition,
    partitioned_vertices,
    partitionedgraph,
    quotient_graph,
    quotient_graph_type,
    QuotientVertex,
    QuotientVertices,
    QuotientVertexVertices,
    quotientvertices
using .LazyNamedDimsArrays: lazy, Mul
using DataGraphs: vertex_data_type, vertex_data, edge_data, get_vertices_data
using DataGraphs.DataGraphsPartitionedGraphsExt

function _TensorNetwork end

struct TensorNetwork{V, VD, UG <: AbstractGraph{V}, Tensors <: AbstractDictionary{V, VD}} <:
    AbstractTensorNetwork{V, VD}
    underlying_graph::UG
    tensors::Tensors
    global @inline function _TensorNetwork(
            underlying_graph::UG, tensors::Tensors
        ) where {V, VD, UG <: AbstractGraph{V}, Tensors <: AbstractDictionary{V, VD}}
        # This assumes the tensor connectivity matches the graph structure.
        return new{V, VD, UG, Tensors}(underlying_graph, tensors)
    end
end
# This assumes the tensor connectivity matches the graph structure.
function TensorNetwork(graph::AbstractGraph, tensors)
    return TensorNetwork(graph, Dictionary(keys(tensors), values(tensors)))
end
function TensorNetwork(graph::AbstractGraph, tensors::AbstractDictionary)
    tn = _TensorNetwork(graph, tensors)
    fix_links!(tn)
    return tn
end

function TensorNetwork{V, VD, UG, Tensors}(graph::UG) where {V, VD, UG <: AbstractGraph{V}, Tensors}
    return _TensorNetwork(graph, Tensors())
end

# DataGraphs interface

DataGraphs.underlying_graph(tn::TensorNetwork) = tn.underlying_graph

DataGraphs.is_vertex_assigned(tn::TensorNetwork, v) = haskey(tn.tensors, v)
DataGraphs.is_edge_assigned(tn::TensorNetwork, e) = false

DataGraphs.get_vertex_data(tn::TensorNetwork, v) = tn.tensors[v]

DataGraphs.set_vertex_data!(tn::TensorNetwork, val, v) = set!(tn.tensors, v, val)

function DataGraphs.underlying_graph_type(type::Type{<:TensorNetwork})
    return fieldtype(type, :underlying_graph)
end

# For a collection of tensors, return the edges implied by shared indices
# as a list of `edgetype` edges of keys/vertices.
function tensornetwork_edges(edgetype::Type, tensors)
    # We need to collect the keys since in the case of `tensors::AbstractDictionary`,
    # `keys(tensors)::AbstractIndices`, which is indexed by `keys(tensors)` rather
    # than `1:length(keys(tensors))`, which is assumed by `combinations`.
    verts = collect(keys(tensors))
    return filter(
        !isnothing, map(combinations(verts, 2)) do (v1, v2)
            if !isdisjoint(inds(tensors[v1]), inds(tensors[v2]))
                return arrange_edge(edgetype(v1, v2))
            end
            return nothing
        end
    )
end
tensornetwork_edges(tensors) = tensornetwork_edges(NamedEdge, tensors)

function TensorNetwork(f::Base.Callable, graph::AbstractGraph)
    return TensorNetwork(graph, Dictionary(map(f, vertices(graph))))
end

# Insert trivial links for missing edges, and also check
# the vertices and edges are consistent between the graph and tensors.
function fix_links!(tn::AbstractTensorNetwork)
    graph = underlying_graph(tn)
    tensors = vertex_data(tn)
    @assert issetequal(vertices(graph), keys(tensors)) "Graph vertices and tensor keys must match."
    tn_edges = tensornetwork_edges(edgetype(graph), tensors)
    tn_edges ⊆ arranged_edges(graph) ||
        error("The edges in the tensors do not match the graph structure.")
    for e in setdiff(arranged_edges(graph), tn_edges)
        insert_trivial_link!(tn, e)
    end
    return tn
end

# Determine the graph structure from the tensors.
function TensorNetwork(tensors)
    graph = NamedGraph(keys(tensors))
    add_edges!(graph, tensornetwork_edges(tensors))
    return _TensorNetwork(graph, tensors)
end

function Base.copy(tn::TensorNetwork)
    return TensorNetwork(copy(underlying_graph(tn)), copy(vertex_data(tn)))
end
TensorNetwork(tn::TensorNetwork) = copy(tn)
TensorNetwork{V}(tn::TensorNetwork{V}) where {V} = copy(tn)
function TensorNetwork{V}(tn::TensorNetwork) where {V}
    g = convert_vertextype(V, underlying_graph(tn))
    d = dictionary(V(k) => tn[k] for k in keys(d))
    return TensorNetwork(g, d)
end

NamedGraphs.convert_vertextype(::Type{V}, tn::TensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::TensorNetwork) = TensorNetwork{V}(tn)

Graphs.connected_components(tn::TensorNetwork) = Graphs.connected_components(underlying_graph(tn))

function Graphs.rem_edge!(tn::TensorNetwork, e)
    if !has_edge(underlying_graph(tn), e)
        return false
    end
    if !isempty(linkinds(tn, e))
        throw(ArgumentError("cannot remove edge $e due to tensor indices existing on this edge."))
    end
    rem_edge!(underlying_graph(tn), e)
    return true
end

function GraphsExtensions.similar_graph(type::Type{<:TensorNetwork})
    DT = fieldtype(type, :tensors)
    empty_dict = DT()
    return TensorNetwork(similar_graph(underlying_graph_type(type)), empty_dict)
end
function GraphsExtensions.similar_graph(tn::TensorNetwork, underlying_graph::AbstractGraph)
    DT = fieldtype(typeof(tn), :tensors)
    empty_dict = DT()
    return _TensorNetwork(underlying_graph, empty_dict)
end

function NamedGraphs.induced_subgraph_from_vertices(graph::TensorNetwork, subvertices)
    return tensornetwork_induced_subgraph(graph, subvertices)
end

function tensornetwork_induced_subgraph(graph, subvertices)
    underlying_subgraph, vlist = Graphs.induced_subgraph(underlying_graph(graph), subvertices)

    subgraph = TensorNetwork(underlying_subgraph) do vertex
        return graph[vertex]
    end

    return subgraph, vlist
end

## PartitionedGraphs
function PartitionedGraphs.quotient_graph(tn::TensorNetwork)
    ug = quotient_graph(underlying_graph(tn))

    inds = Indices(parent_graph_indices(QuotientVertices(tn)))
    data = map(v -> tn[QuotientVertex(v)], inds)

    return TensorNetwork(ug, data)
end
# TODO: This method should not be required with a better interface with a better
# DataGraphsPartitionedGraphsExt interface.
function PartitionedGraphs.quotient_graph_type(type::Type{<:TensorNetwork})
    UG = quotient_graph_type(underlying_graph_type(type))
    VD = Vector{vertex_data_type(type)}
    V = vertextype(UG)
    return TensorNetwork{V, VD, UG, Dictionary{V, VD}}
end

# Partition the underlying graph of the tensor network; does not affect the data.
function PartitionedGraphs.partitionedgraph(tn::TensorNetwork, parts)
    pg = partitionedgraph(underlying_graph(tn), parts)
    return TensorNetwork(pg, copy(vertex_data(tn)))
end

PartitionedGraphs.departition(tn::TensorNetwork) = tn
function PartitionedGraphs.departition(
        tn::TensorNetwork{<:Any, <:Any, <:AbstractPartitionedGraph}
    )
    return TensorNetwork(departition(underlying_graph(tn)), vertex_data(tn))
end

NamedGraphs.to_graph_index(::TensorNetwork, vertex::QuotientVertex) = vertex
# When getting data according the quotient vertices, take a lazy contraction.
function DataGraphs.get_index_data(tn::TensorNetwork, vertex::QuotientVertex)
    data = collect(map(v -> tn[v], vertices(tn, vertex)))
    return mapreduce(lazy, *, data)
end
function DataGraphs.is_graph_index_assigned(tn::TensorNetwork, vertex::QuotientVertex)
    return isassigned(tn, Vertices(vertices(tn, vertex)))
end

function PartitionedGraphs.quotientview(tn::TensorNetwork)
    qview = QuotientView(underlying_graph(tn))
    tensors = map(qv -> vertex_data(tn)[Indices(qv)], Indices(quotientvertices(tn)))
    return TensorNetwork(qview, tensors)
end
