struct KetView{T, V, I} <: AbstractTensorNetwork{T, V}
    parent::NormNetwork{T, V, I}
end

struct BraView{T, V, I} <: AbstractTensorNetwork{T, V}
    parent::NormNetwork{T, V, I}
end

# ====================================== Graphs.jl ======================================= #

for View in (:KetView, :BraView)
    @eval begin
        Graphs.edges(nnv::$View) = edges(nnv.parent.tensornetwork)
        Graphs.vertices(nnv::$View) = vertices(nnv.parent.tensornetwork)
    end
end

# ==================================== NamedGraphs.jl ==================================== #

for View in (:KetView, :BraView)
    @eval begin
        function NamedGraphs.vertex_positions(nnv::$View)
            return index_positions(vertices(nnv))
        end
        function NamedGraphs.ordered_vertices(nnv::$View)
            return ordered_indices(vertices(nnv))
        end

        NamedGraphs.position_graph(nnv::$View) = position_graph(nnv.parent.tensornetwork)
    end
end

# ==================================== DataGraphs.jl ===================================== #

DataGraphs.get_vertex_data(nn::KetView, vertex) = ket(nn.parent, vertex)
DataGraphs.get_vertex_data(nn::BraView, vertex) = bra(nn.parent, vertex)

for View in (:KetView, :BraView)
    @eval begin
        function DataGraphs.is_vertex_assigned(nnv::$View, vertex)
            return isassigned(nnv.parent.tensornetwork, vertex)
        end
    end
end

# =================================== Dictionaries.jl ==================================== #

for View in (:KetView, :BraView)
    @eval begin
        Dictionaries.issettable(nnv::$View) = issettable(nnv.parent)
        Dictionaries.isinsertable(::$View) = isinsertable(nnv.parent)
    end
end
