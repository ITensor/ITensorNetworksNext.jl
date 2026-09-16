using DataGraphs: DataGraphs, is_vertex_assigned
using Dictionaries: Dictionaries, isinsertable, issettable
using Graphs: Graphs, edges, vertices
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

"""
    abstract type AbstractBilinearFormNetworkView{T, V, I} <: AbstractITensorNetwork{T, V}

Supertype of the single-layer views of an `AbstractBilinearFormNetwork{T, V, I}`.

A subtype implements `Base.parent`, returning the network it views, and
`DataGraphs.get_vertex_data`, returning that layer's tensor at a vertex. Its graph structure
and mutability are those of the parent network.
"""
abstract type AbstractBilinearFormNetworkView{T, V, I} <: AbstractITensorNetwork{T, V} end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(nnv::AbstractBilinearFormNetworkView) = edges(parent(nnv))
Graphs.vertices(nnv::AbstractBilinearFormNetworkView) = vertices(parent(nnv))

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.encoded_vertex(nnv::AbstractBilinearFormNetworkView, vertex)
    return encoded_vertex(parent(nnv), vertex)
end
function NamedGraphs.decoded_vertex(nnv::AbstractBilinearFormNetworkView, code::Integer)
    return decoded_vertex(parent(nnv), code)
end
NamedGraphs.encoded_graph(nnv::AbstractBilinearFormNetworkView) = encoded_graph(parent(nnv))

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.is_vertex_assigned(nnv::AbstractBilinearFormNetworkView, vertex)
    return is_vertex_assigned(parent(nnv), vertex)
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(nnv::AbstractBilinearFormNetworkView) = issettable(parent(nnv))
Dictionaries.isinsertable(nnv::AbstractBilinearFormNetworkView) = isinsertable(parent(nnv))
