using DataGraphs: DataGraphs, get_vertex_data, is_vertex_assigned
using Dictionaries: Dictionaries, isinsertable, issettable
using Graphs: Graphs, edges, vertices

"""
    struct BraView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <: AbstractITensorNetwork{T, V}

The bra layer of the bilinear-form network `parent(view)`, with each vertex tensor built by
[`bratensor`](@ref) when accessed. Its graph structure and mutability are those of the parent.
"""
struct BraView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <:
    AbstractITensorNetwork{T, V}
    parent::P
    function BraView(parent::AbstractBilinearFormNetwork{T, V, I}) where {T, V, I}
        return new{T, V, I, typeof(parent)}(parent)
    end
end

Base.parent(nnv::BraView) = nnv.parent

# ==================================== DataGraphs.jl ===================================== #

DataGraphs.get_vertex_data(nnv::BraView, vertex) = bratensor(parent(nnv), vertex)
