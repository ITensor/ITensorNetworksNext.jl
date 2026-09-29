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
        return new{T, V, I}(ket, select_branames(ket, map, ()))
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
branamemap(g::NormGramian) = g.braname
layertensors(g::NormGramian) = (; ket = kettensor(g), bra = bratensor(g))
layerinds(g::NormGramian) = (inds(kettensor(g)), brainds(g))

Base.eltype(::Type{<:NormNetwork{T, V, I}}) where {T, V, I} = NormGramian{T, I}

function NormNetwork(tn::ITensorNetwork)
    return NormNetwork(tn, map(uniquename, keys(tn.dimname_vertices)))
end

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(nn::NormNetwork{T, V, I}, vertex) where {T, V, I}
    return NormGramian{T, I}(nn.ket[vertex], nn.braname)
end

# ====================================== interface ======================================= #

ketnetwork(nn::NormNetwork) = nn.ket
branamemap(nn::NormNetwork) = nn.braname

"""
    normnetwork(tn::ITensorNetwork, [braname]) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NomnNetwork` object.
The optional second argument `braname` should implement `braname[ketdimname] = bradimname` for
every link dimension name `ketdimname` in `tn`. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
normnetwork(tn::ITensorNetwork) = NormNetwork(tn)
normnetwork(tn::ITensorNetwork, braname) = NormNetwork(tn, braname)
