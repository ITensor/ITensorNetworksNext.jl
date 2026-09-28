using Dictionaries: Dictionary
using ITensorBase: similar_operator, uniquename
using ITensorNetworksNext

"""
    struct NormNetwork{T, V, I} <: AbstractBilinearFormNetwork{T, V, I}

Lazy wrapper representing the norm `⟨tn|tn⟩` of `tn::ITensorNetwork{T, V, I}`,
together with a per-edge ket→bra name mapping that, for each index in the ket layer, defines
the name of the corresponding index in the bra layer.
"""
struct NormNetwork{T, V, I} <: AbstractBilinearFormNetwork{T, V, I}
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

"""
    struct NormGramian{T, I} <: AbstractGramian

The layers of a `NormNetwork` at one vertex: the ket tensor and the network's ket→bra name map,
from which the bra tensor is built when requested.
"""
struct NormGramian{T, I} <: AbstractGramian
    ket::T
    braname::Dictionary{I, I}
end

kettensor(g::NormGramian) = g.ket
braname(g::NormGramian, name) = get(g.braname, name, name)
layertensors(g::NormGramian) = (; ket = kettensor(g), bra = bratensor(g))

Base.eltype(::Type{<:NormNetwork{T, V, I}}) where {T, V, I} = NormGramian{T, I}

function NormNetwork(tn::ITensorNetwork)
    return NormNetwork(tn, map(uniquename, keys(tn.dimname_vertices)))
end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(nn::NormNetwork) = edges(nn.ket)
Graphs.vertices(nn::NormNetwork) = vertices(nn.ket)

# ==================================== NamedGraphs.jl ==================================== #

NamedGraphs.encoded_vertex(nn::NormNetwork, vertex) = encoded_vertex(nn.ket, vertex)
NamedGraphs.decoded_vertex(nn::NormNetwork, code::Integer) = decoded_vertex(nn.ket, code)
NamedGraphs.encoded_graph(nn::NormNetwork) = encoded_graph(nn.ket)

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.is_vertex_assigned(nn::NormNetwork, vertex)
    return isassigned(nn.ket, vertex)
end

function DataGraphs.get_vertex_data(nn::NormNetwork{T, V, I}, vertex) where {T, V, I}
    return NormGramian{T, I}(nn.ket[vertex], nn.braname)
end

# ====================================== interface ======================================= #

function braname(nn::NormNetwork, name)
    if !has_dimname(nn.ket, name)
        error("index name $name not found underlying tensor network.")
    end
    # The indices not stored in `nn.braname` are precisely the site indices, which
    # get mapped to themselves.
    return get(nn.braname, name, name)
end

ketnetwork(nn::NormNetwork) = nn.ket

"""
    normnetwork(tn::ITensorNetwork, [braname]) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NomnNetwork` object.
The optional second argument `braname` should implement `braname[ketdimname] = bradimname` for
every link dimension name `ketdimname` in `tn`. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
normnetwork(tn::ITensorNetwork) = NormNetwork(tn)
normnetwork(tn::ITensorNetwork, braname) = NormNetwork(tn, braname)
