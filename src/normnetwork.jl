using Dictionaries: Dictionary
using ITensorBase: LazyNamedTensor, lazy, replacedimnames, setname, similar_operator, uniquename
using ITensorNetworksNext

"""
    struct NormNetwork{T, V, I} <: AbstractITensorNetwork{T, V}

Lazy wrapper representing the norm `⟨tn|tn⟩` of `tn::ITensorNetwork{T, V, I}`,
together with a per-edge ket→bra name mapping that, for each index in the ket layer, defines
the name of the corresponding index in the bra layer.
"""
struct NormNetwork{T, V, I} <: AbstractITensorNetwork{T, V}
    ket::ITensorNetwork{T, V, I}
    braname::Dictionary{I, I}
    function NormNetwork(
            ket::ITensorNetwork{T, V, I},
            map::Dictionary{I, I}
        ) where {T, V, I}
        braname = Dictionary{I, I}()
        for (name, vertices) in pairs(ket.dimname_vertices)
            if length(vertices) == 2
                insert!(braname, name, map[name])
            end
        end
        return new{T, V, I}(ket, braname)
    end
end

Base.eltype(::Type{<:NormNetwork{T, V, I}}) where {T, V, I} = LazyNamedTensor{I, T}

function NormNetwork(tn::ITensorNetwork)
    return NormNetwork(tn, map(uniquename, keys(tn.dimname_vertices)))
end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(nn::NormNetwork) = edges(nn.ket)
Graphs.vertices(nn::NormNetwork) = vertices(nn.ket)

# ==================================== NamedGraphs.jl ==================================== #

NamedGraphs.vertex_positions(nn::NormNetwork) = vertex_positions(nn.ket)
NamedGraphs.ordered_vertices(nn::NormNetwork) = ordered_vertices(nn.ket)
NamedGraphs.position_graph(nn::NormNetwork) = position_graph(nn.ket)

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(nn::NormNetwork, vertex)
    A = kettensor(nn, vertex)
    B = conj_bratensor(nn, vertex)
    # TODO: implement and use a lazy `conj` via `LazyNamedDimsArrays` here?
    return lazy(A) * lazy(conj(B))
end

function DataGraphs.is_vertex_assigned(nn::NormNetwork, vertex)
    return isassigned(nn.ket, vertex)
end
# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::NormNetwork) = false
Dictionaries.isinsertable(::NormNetwork) = false

# ====================================== interface ======================================= #

function braname(nn::NormNetwork, name)
    if !has_dimname(nn.ket, name)
        error("index name $name not found underlying tensor network.")
    end
    # The indices not stored in `nn.braname` are precisely the site indices, which
    # get mapped to themselves.
    return get(nn.braname, name, name)
end

indmap(nn::NormNetwork, ind) = setname(conj(ind), braname(nn, name(ind)))

kettensor(nn::NormNetwork, vertex) = nn.ket[vertex]
function conj_bratensor(nn::NormNetwork, vertex)
    return replacedimnames(n -> braname(nn, n), kettensor(nn, vertex))
end

bratensor(nn::NormNetwork, vertex) = conj(conj_bratensor(nn, vertex))

"""
    normnetwork(tn::ITensorNetwork, [braname]) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NomnNetwork` object.
The optional second argument `braname` should implement `braname[ketdimname] = bradimname` for
every link dimension name `ketdimname` in `tn`. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
normnetwork(tn::ITensorNetwork) = NormNetwork(tn)
normnetwork(tn::ITensorNetwork, braname) = NormNetwork(tn, braname)
