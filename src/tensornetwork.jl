using Combinatorics: combinations
using DataGraphs.DataGraphsPartitionedGraphsExt
using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph, edge_data, get_vertices_data,
    vertex_data, vertex_data_type
using Dictionaries: AbstractDictionary, Indices, dictionary, set!, unset!
using Graphs: AbstractSimpleGraph, rem_edge!, rem_vertex!
using ITensorBase: AbstractITensor, dimnames, lazy
using NamedGraphs.GraphsExtensions:
    GraphsExtensions, arrange_edge, arranged_edges, vertextype
using NamedGraphs.PartitionedGraphs: AbstractPartitionedGraph, PartitionedGraphs,
    QuotientVertex, QuotientVertexVertices, QuotientVertices, departition,
    partitioned_vertices, partitionedgraph, quotient_graph, quotient_graph_type,
    quotientvertices
using NamedGraphs:
    NamedGraphs, NamedEdge, NamedGraph, Vertices, parent_graph_indices, vertextype

function _TensorNetwork end

struct ITensorNetwork{
        V,
        VD,
        UG <: AbstractGraph{V},
        Tensors <: AbstractDictionary{V, VD},
    } <:
    AbstractITensorNetwork{V, VD}
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
function ITensorNetwork(graph::AbstractGraph, tensors)
    return ITensorNetwork(graph, Dictionary(keys(tensors), values(tensors)))
end
function ITensorNetwork(graph::AbstractGraph, tensors::AbstractDictionary)
    tn = _TensorNetwork(graph, tensors)
    fix_links!(tn)
    return tn
end

function ITensorNetwork{V, VD, UG, Tensors}(
        graph::UG
    ) where {V, VD, UG <: AbstractGraph{V}, Tensors}
    return _TensorNetwork(graph, Tensors())
end

function Graphs.rem_vertex!(tn::ITensorNetwork, v)
    delete!(tn.tensors, v)
    rem_vertex!(tn.underlying_graph, v)
    return tn
end

# DataGraphs interface

DataGraphs.underlying_graph(tn::ITensorNetwork) = tn.underlying_graph

DataGraphs.is_vertex_assigned(tn::ITensorNetwork, v) = haskey(tn.tensors, v)
DataGraphs.is_edge_assigned(tn::ITensorNetwork, e) = false

DataGraphs.get_vertex_data(tn::ITensorNetwork, v) = tn.tensors[v]

DataGraphs.set_vertex_data!(tn::ITensorNetwork, val, v) = set!(tn.tensors, v, val)

function DataGraphs.underlying_graph_type(type::Type{<:ITensorNetwork})
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
            if !isdisjoint(dimnames(tensors[v1]), dimnames(tensors[v2]))
                return arrange_edge(edgetype(v1, v2))
            end
            return nothing
        end
    )
end
tensornetwork_edges(tensors) = tensornetwork_edges(NamedEdge, tensors)

function ITensorNetwork(f::Base.Callable, graph::AbstractGraph)
    return ITensorNetwork(graph, Dictionary(map(f, vertices(graph))))
end

# Insert trivial links for missing edges, and also check
# the vertices and edges are consistent between the graph and tensors.
function fix_links!(tn::AbstractITensorNetwork)
    graph = underlying_graph(tn)
    tensors = vertex_data(tn)
    @assert issetequal(vertices(graph), keys(tensors)) "Graph vertices and tensor keys must match."
    tn_edges = tensornetwork_edges(edgetype(graph), tensors)
    tn_edges ⊆ arranged_edges(graph) ||
        error("The edges in the tensors do not match the graph structure.")
    for e in setdiff(arranged_edges(graph), tn_edges)
        insertlink!(tn, e)
    end
    return tn
end

# Determine the graph structure from the tensors.
function ITensorNetwork(tensors)
    graph = NamedGraph(keys(tensors))
    add_edges!(graph, tensornetwork_edges(tensors))
    return _TensorNetwork(graph, tensors)
end

function Base.copy(tn::ITensorNetwork)
    return ITensorNetwork(copy(underlying_graph(tn)), copy(vertex_data(tn)))
end
ITensorNetwork(tn::ITensorNetwork) = copy(tn)
ITensorNetwork{V}(tn::ITensorNetwork{V}) where {V} = copy(tn)
function ITensorNetwork{V}(tn::ITensorNetwork) where {V}
    g = convert_vertextype(V, underlying_graph(tn))
    d = dictionary(V(k) => tn[k] for k in vertices(tn))
    return ITensorNetwork(g, d)
end

NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::ITensorNetwork) = ITensorNetwork{V}(tn)

function Graphs.rem_edge!(tn::ITensorNetwork, e)
    if !has_edge(underlying_graph(tn), e)
        return false
    end
    if !isempty(linknames(tn, e))
        throw(
            ArgumentError(
                "cannot remove edge $e due to tensor indices existing on this edge."
            )
        )
    end
    rem_edge!(underlying_graph(tn), e)
    return true
end

function NamedGraphs.similar_graph(
        type::Type{<:ITensorNetwork},
        vertices = vertextype(type)[]
    )
    DT = fieldtype(type, :tensors)
    empty_dict = DT()

    underlying_graph = similar_graph(underlying_graph_type(type), vertices)

    return _TensorNetwork(underlying_graph, empty_dict)
end
function NamedGraphs.similar_graph(
        graph::ITensorNetwork,
        VD::Type,
        ::Type{<:Nothing},
        vertices
    )
    V = eltype(vertices)
    empty_dict = Dictionary{V, VD}()

    new_underlying_graph = similar_graph(underlying_graph(graph), vertices)

    return _TensorNetwork(new_underlying_graph, empty_dict)
end

function NamedGraphs.induced_subgraph_from_vertices(graph::ITensorNetwork, subvertices)
    return induced_subgraph_tensornetwork(graph, subvertices)
end

function induced_subgraph_tensornetwork(graph, subvertices)
    underlying_subgraph, vlist =
        Graphs.induced_subgraph(underlying_graph(graph), subvertices)

    subgraph = ITensorNetwork(underlying_subgraph) do vertex
        return graph[vertex]
    end

    return subgraph, vlist
end

## PartitionedGraphs
function PartitionedGraphs.partitioned_vertices(tn::ITensorNetwork)
    return partitioned_vertices(tn.underlying_graph)
end

function PartitionedGraphs.quotient_graph(tn::ITensorNetwork)
    ug = quotient_graph(underlying_graph(tn))

    inds = Indices(parent_graph_indices(QuotientVertices(tn)))
    data = map(v -> tn[QuotientVertex(v)], inds)

    return ITensorNetwork(ug, data)
end
# TODO: This method should not be required with a better interface with a better
# DataGraphsPartitionedGraphsExt interface.
function PartitionedGraphs.quotient_graph_type(type::Type{<:ITensorNetwork})
    UG = quotient_graph_type(underlying_graph_type(type))
    VD = Vector{vertex_data_type(type)}
    V = vertextype(UG)
    return ITensorNetwork{V, VD, UG, Dictionary{V, VD}}
end

# Partition the underlying graph of the tensor network; does not affect the data.
function PartitionedGraphs.partitionedgraph(tn::ITensorNetwork, parts)
    pg = partitionedgraph(underlying_graph(tn), parts)
    return ITensorNetwork(pg, copy(vertex_data(tn)))
end

PartitionedGraphs.departition(tn::ITensorNetwork) = tn
function PartitionedGraphs.departition(
        tn::ITensorNetwork{<:Any, <:Any, <:AbstractPartitionedGraph}
    )
    return ITensorNetwork(departition(underlying_graph(tn)), vertex_data(tn))
end

NamedGraphs.to_graph_index(::ITensorNetwork, vertex::QuotientVertex) = vertex
# When getting data according the quotient vertices, take a lazy contraction.
function DataGraphs.get_index_data(tn::ITensorNetwork, vertex::QuotientVertex)
    data = collect(map(v -> tn[v], vertices(tn, vertex)))
    return mapreduce(lazy, *, data)
end
function DataGraphs.is_graph_index_assigned(tn::ITensorNetwork, vertex::QuotientVertex)
    return isassigned(tn, Vertices(vertices(tn, vertex)))
end

function PartitionedGraphs.quotientview(tn::ITensorNetwork)
    qview = QuotientView(underlying_graph(tn))
    tensors = map(qv -> vertex_data(tn)[Indices(qv)], Indices(quotientvertices(tn)))
    return ITensorNetwork(qview, tensors)
end
