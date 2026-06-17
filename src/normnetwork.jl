using Dictionaries: Dictionary
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy
using NamedDimsArrays: randname, replacedimnames, setname

struct NormNetwork{T, V, I} <: AbstractTensorNetwork{T, V}
    tensornetwork::TensorNetwork{T, V, I}
    namemap::Dictionary{I, I}
    function NormNetwork(tn::TensorNetwork{T, V, I}, map::Dictionary{I, I}) where {T, V, I}
        namemap = Dictionary{I, I}()
        for (name, vertices) in pairs(tn.index_locations)
            if length(vertices) == 2
                insert!(namemap, name, map[name])
            end
        end
        return new{T, V, I}(tn, namemap)
    end
end

Base.eltype(::Type{<:NormNetwork{T}}) where {T} = LazyNamedDimsArray{eltype(T), T}

NormNetwork(tn::TensorNetwork) = NormNetwork(tn, map(randname, keys(tn.index_locations)))

# ====================================== Graphs.jl ======================================= #

Graphs.edges(nn::NormNetwork) = edges(nn.tensornetwork)
Graphs.vertices(nn::NormNetwork) = vertices(nn.tensornetwork)

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.vertex_positions(nn::NormNetwork)
    return index_positions(vertices(nn))
end
function NamedGraphs.ordered_vertices(nn::NormNetwork)
    return ordered_indices(vertices(nn))
end

NamedGraphs.position_graph(nn::NormNetwork) = position_graph(nn.tensornetwork)

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(nn::NormNetwork, vertex)
    A = ket(nn, vertex)
    B = conjbra(nn, vertex)
    # TODO: implement and use a lazy `conj` via `LazyNamedDimsArrays` here?
    return lazy(A) * lazy(conj(B))
end

function DataGraphs.is_vertex_assigned(nn::NormNetwork, vertex)
    return isassigned(nn.tensornetwork, vertex)
end
# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::NormNetwork) = false
Dictionaries.isinsertable(::NormNetwork) = false

# ====================================== interface ======================================= #

tensornetwork(nn::NormNetwork) = nn.tensornetwork

function namemap(nn::NormNetwork, name)
    if !has_indname(nn.tensornetwork, name)
        error("index name $name not found underlying tensor network.")
    end
    return get(nn.namemap, name, name)
end
indmap(nn::NormNetwork, ind) = setname(ind, namemap(nn, name(ind)))

ket(nn::NormNetwork, vertex) = nn.tensornetwork[vertex]
conjbra(nn::NormNetwork, vertex) = replacedimnames(n -> namemap(nn, n), ket(nn, vertex))

lazy_norm(tn::TensorNetwork) = NormNetwork(tn)
