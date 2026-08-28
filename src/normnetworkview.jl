using DataGraphs: DataGraphs, get_vertex_data, is_vertex_assigned
using Dictionaries: Dictionaries, isinsertable, issettable
using Graphs: Graphs, edges, vertices
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

struct KetView{T, V, I} <: AbstractITensorNetwork{T, V}
    parent::NormNetwork{T, V, I}
end

struct BraView{T, V, I} <: AbstractITensorNetwork{T, V}
    parent::NormNetwork{T, V, I}
end

# ====================================== Graphs.jl ======================================= #

for View in (:KetView, :BraView)
    @eval begin
        Graphs.edges(nnv::$View) = edges(nnv.parent)
        Graphs.vertices(nnv::$View) = vertices(nnv.parent)
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

# ==================================== DataGraphs.jl ===================================== #

DataGraphs.get_vertex_data(nn::KetView, vertex) = kettensor(nn.parent, vertex)
DataGraphs.get_vertex_data(nn::BraView, vertex) = bratensor(nn.parent, vertex)

for View in (:KetView, :BraView)
    @eval begin
        function DataGraphs.is_vertex_assigned(nnv::$View, vertex)
            return isassigned(nnv.parent.ket, vertex)
        end
    end
end

# =================================== Dictionaries.jl ==================================== #

for View in (:KetView, :BraView)
    @eval begin
        Dictionaries.issettable(nnv::$View) = issettable(nnv.parent)
        Dictionaries.isinsertable(nnv::$View) = isinsertable(nnv.parent)
    end
end
