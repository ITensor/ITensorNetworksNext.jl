using DataGraphs: DataGraphs, get_vertex_data, is_vertex_assigned
using Dictionaries: Dictionaries, isinsertable, issettable
using Graphs: Graphs, edges, vertices
using ITensorBase: ITensorBase, conj, inds
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

"""
    struct BraView{T, V, I} <: AbstractITensorNetwork{T, V}

The bra layer of the norm network `parent(view)`, with each vertex tensor built by
[`bratensor`](@ref) when accessed. Its graph structure and mutability are those of the parent.
"""
struct BraView{T, V, I} <: AbstractITensorNetwork{T, V}
    parent::NormNetwork{T, V, I}
end

Base.parent(nnv::BraView) = nnv.parent

# ==================================== DataGraphs.jl ===================================== #

DataGraphs.get_vertex_data(nnv::BraView, vertex) = bratensor(parent(nnv), vertex)
function DataGraphs.is_vertex_assigned(nnv::BraView, vertex)
    return is_vertex_assigned(parent(nnv), vertex)
end

# ==================================== ITensorBase.jl ==================================== #

function ITensorBase.inds(nnv::BraView, vertex)
    return conj.(inds(conj_bratensor(parent(nnv), vertex)))
end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(nnv::BraView) = edges(parent(nnv))
Graphs.vertices(nnv::BraView) = vertices(parent(nnv))

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.encoded_vertex(nnv::BraView, vertex)
    return encoded_vertex(parent(nnv), vertex)
end
function NamedGraphs.decoded_vertex(nnv::BraView, code::Integer)
    return decoded_vertex(parent(nnv), code)
end
NamedGraphs.encoded_graph(nnv::BraView) = encoded_graph(parent(nnv))

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(nnv::BraView) = issettable(parent(nnv))
Dictionaries.isinsertable(nnv::BraView) = isinsertable(parent(nnv))
