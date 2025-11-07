using Combinatorics: combinations
using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph
using Dictionaries: AbstractDictionary, Indices, dictionary
using Graphs: AbstractSimpleGraph
using NamedDimsArrays: AbstractNamedDimsArray, dimnames
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype
using NamedGraphs.GraphsExtensions: add_edges!, arrange_edge, arranged_edges, vertextype

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
function _TensorNetwork(graph::AbstractGraph, tensors)
    return _TensorNetwork(graph, Dictionary(keys(tensors), values(tensors)))
end

DataGraphs.underlying_graph(tn::TensorNetwork) = getfield(tn, :underlying_graph)
DataGraphs.vertex_data(tn::TensorNetwork) = getfield(tn, :tensors)
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
    tensors = Dictionary(vertices(graph), f.(vertices(graph)))
    return TensorNetwork(graph, tensors)
end
function TensorNetwork(graph::AbstractGraph, tensors)
    tn = _TensorNetwork(graph, tensors)
    fix_links!(tn)
    return tn
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
