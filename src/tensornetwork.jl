using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph
using Dictionaries: AbstractDictionary, Indices, dictionary
using Graphs: AbstractSimpleGraph
using NamedDimsArrays: AbstractNamedDimsArray, dimnames
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype
using NamedGraphs.GraphsExtensions: arranged_edges, vertextype

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

DataGraphs.underlying_graph(tn::TensorNetwork) = getfield(tn, :underlying_graph)
DataGraphs.vertex_data(tn::TensorNetwork) = getfield(tn, :tensors)
function DataGraphs.underlying_graph_type(type::Type{<:TensorNetwork})
    return fieldtype(type, :underlying_graph)
end

# Determine the graph structure from the tensors.
function TensorNetwork(t::AbstractDictionary)
    g = NamedGraph(eachindex(t))
    for v1 in vertices(g)
        for v2 in vertices(g)
            if v1 ≠ v2
                if !isdisjoint(dimnames(t[v1]), dimnames(t[v2]))
                    add_edge!(g, v1 => v2)
                end
            end
        end
    end
    return _TensorNetwork(g, t)
end
function TensorNetwork(tensors::AbstractDict)
    return TensorNetwork(Dictionary(tensors))
end

function TensorNetwork(graph::AbstractGraph, tensors::AbstractDictionary)
    tn = TensorNetwork(tensors)
    arranged_edges(tn) ⊆ arranged_edges(graph) ||
        error("The edges in the tensors do not match the graph structure.")
    for e in setdiff(arranged_edges(graph), arranged_edges(tn))
        insert_trivial_link!(tn, e)
    end
    return tn
end
function TensorNetwork(graph::AbstractGraph, tensors::AbstractDict)
    return TensorNetwork(graph, Dictionary(tensors))
end
function TensorNetwork(f, graph::AbstractGraph)
    return TensorNetwork(graph, Dict(v => f(v) for v in vertices(graph)))
end

function Base.copy(tn::TensorNetwork)
    return TensorNetwork(copy(underlying_graph(tn)), copy(vertex_data(tn)))
end
TensorNetwork(tn::TensorNetwork) = copy(tn)
TensorNetwork{V}(tn::TensorNetwork{V}) where {V} = copy(tn)
function TensorNetwork{V}(tn::TensorNetwork) where {V}
    g′ = convert_vertextype(V, underlying_graph(tn))
    d = vertex_data(tn)
    d′ = dictionary(V(k) => d[k] for k in eachindex(d))
    return TensorNetwork(g′, d′)
end

NamedGraphs.convert_vertextype(::Type{V}, tn::TensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::TensorNetwork) = TensorNetwork{V}(tn)
