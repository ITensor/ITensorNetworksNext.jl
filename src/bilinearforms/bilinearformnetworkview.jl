using DataGraphs: DataGraphs, get_vertex_data, is_vertex_assigned
using Dictionaries: Dictionaries, isinsertable, issettable
using Graphs: Graphs, edges, vertices
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

struct KetView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <:
    AbstractBilinearFormNetworkView{T, V, I}
    parent::P
    function KetView(parent::AbstractBilinearFormNetwork{T, V, I}) where {T, V, I}
        return new{T, V, I, typeof(parent)}(parent)
    end
end

# ==================================== NamedGraphs.jl ==================================== #

for View in (:KetView, :BraView)
    @eval begin
        function NamedGraphs.encoded_vertex(nnv::$View, vertex)
            return encoded_vertex(nnv.parent, vertex)
        end
        function NamedGraphs.decoded_vertex(nnv::$View, code::Integer)
            return decoded_vertex(nnv.parent, code)
        end

        NamedGraphs.encoded_graph(nnv::$View) = encoded_graph(nnv.parent)
    end
end

struct BraView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <:
    AbstractBilinearFormNetworkView{T, V, I}
    parent::P
    function BraView(parent::AbstractBilinearFormNetwork{T, V, I}) where {T, V, I}
        return new{T, V, I, typeof(parent)}(parent)
    end
end

struct OperatorView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <:
    AbstractBilinearFormNetworkView{T, V, I}
    parent::P
    function OperatorView(parent::AbstractBilinearFormNetwork{T, V, I}) where {T, V, I}
        return new{T, V, I, typeof(parent)}(parent)
    end
end

# A `NormNetwork`'s bra and ket layers share their site index names, so there is no distinct
# bra site index for an identity operator to carry as its output name.
function OperatorView(::NormNetwork)
    throw(ArgumentError("a `NormNetwork` has no operator layer to view."))
end

Base.parent(nnv::KetView) = nnv.parent
Base.parent(nnv::BraView) = nnv.parent
Base.parent(nnv::OperatorView) = nnv.parent

# ==================================== DataGraphs.jl ===================================== #

DataGraphs.get_vertex_data(nnv::KetView, vertex) = kettensor(parent(nnv), vertex)
DataGraphs.get_vertex_data(nnv::BraView, vertex) = bratensor(parent(nnv), vertex)
DataGraphs.get_vertex_data(nnv::OperatorView, vertex) = operatortensor(parent(nnv), vertex)
