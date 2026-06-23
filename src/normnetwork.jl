using Dictionaries: Dictionary
using ITensorBase: LazyITensor, lazy, replacedimnames, setname, similar_operator, uniquename
using ITensorNetworksNext

"""
    struct NormNetwork{T, V, I} <: AbstractTensorNetwork{T, V}

Lazy wrapper representing the norm `⟨tn|tn⟩` of `tn::TensorNetwork{T, V, I}`,
together with a per-edge ket→bra name mapping that, for each index in the ket layer, defines
the name of the corresponding index in the bra layer.
"""
struct NormNetwork{T, V, I} <: AbstractTensorNetwork{T, V}
    tensornetwork::TensorNetwork{T, V, I}
    namemap::Dictionary{I, I}
    function NormNetwork(tn::TensorNetwork{T, V, I}, map::Dictionary{I, I}) where {T, V, I}
        namemap = Dictionary{I, I}()
        for (name, vertices) in pairs(tn.dimname_vertices)
            if length(vertices) == 2
                insert!(namemap, name, map[name])
            end
        end
        return new{T, V, I}(tn, namemap)
    end
end

Base.eltype(::Type{<:NormNetwork{T}}) where {T} = LazyITensor{eltype(T), T}

NormNetwork(tn::TensorNetwork) = NormNetwork(tn, map(uniquename, keys(tn.dimname_vertices)))

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

indmap(nn::NormNetwork, ind) = setname(conj(ind), namemap(nn, name(ind)))

ket(nn::NormNetwork, vertex) = nn.tensornetwork[vertex]
conjbra(nn::NormNetwork, vertex) = replacedimnames(n -> namemap(nn, n), ket(nn, vertex))

bra(nn::NormNetwork, vertex) = conj(conjbra(nn, vertex))

"""
    normnetwork(tn::TensorNetwork, [namemap]) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NomnNetwork` object.
The optional second argument `namemap` should implement `namemap[ketdimname] = bradimname` for
every link dimension name `ketdimname` in `tn`. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
normnetwork(tn::TensorNetwork) = NormNetwork(tn)
normnetwork(tn::TensorNetwork, namemap) = NormNetwork(tn, namemap)
