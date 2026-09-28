using DataGraphs: DataGraphs, is_vertex_assigned
using Dictionaries: Dictionaries, isinsertable, issettable
using Graphs: Graphs, edges, vertices
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

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

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.is_vertex_assigned(nnv::BraView, vertex)
    return is_vertex_assigned(parent(nnv), vertex)
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(nnv::BraView) = issettable(parent(nnv))
Dictionaries.isinsertable(nnv::BraView) = isinsertable(parent(nnv))
