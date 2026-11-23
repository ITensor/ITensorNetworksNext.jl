using Dictionaries: Dictionary, Indices, dictionary
using ITensorBase: similar_operator
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
        check_input(NormNetwork, ket, map)
        return new{T, V, I}(ket, map)
    end
end

function check_input(::Type{<:NormNetwork}, ket, braname)
    return check_branamemap(ket, braname)
end

function Base.eltype(::Type{<:NormNetwork})
    return error(
        "`eltype` of a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(nn::NormNetwork, vertex)
    return error(
        "Indexing a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

# ====================================== interface ======================================= #

ketnetwork(nn::NormNetwork) = nn.ket
branamemap(nn::NormNetwork) = nn.braname

# The default ket→bra name map
derive_braname(name) = prime(name)

"""
    flatten_network(nn::NormNetwork) -> ITensorNetwork

Expand a norm network into a plain tensor network carrying one vertex per layer, so that vertex
`v` of `nn` becomes the two vertices `(v, :ket)` and `(v, :bra)` and `nv` doubles. A vertex's
two layers stay adjacent in the vertex order, ket first.
"""
function flatten_network(nn::NormNetwork)
    return ITensorNetwork(
        dictionary(
            Iterators.flatten(
                ((v, :ket) => kettensor(nn, v), (v, :bra) => bratensor(nn, v))
                    for v in vertices(nn)
            )
        )
    )
end

"""
    normnetwork([f::Function], tn::ITensorNetwork, [dangling]) -> NormNetwork
    normnetwork(tn::ITensorNetwork, braname::Dictionary) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NormNetwork` object.
Each ket index name in `union(dangling, eachlinkname(tn))` gets the bra name `f(name)`,
where `f` defaults to `derive_braname`, which primes the name, and `dangling` is empty by
default.
Alternatively pass the ket→bra name map `braname` directly, which must include every link
name of `tn` and may include others. Either way an error is thrown if a link name is missing,
two bra names coincide or a bra name is already an index name of `tn`.
"""
normnetwork(tn::ITensorNetwork, braname::Dictionary) = NormNetwork(tn, braname)
function normnetwork(f::Function, tn::ITensorNetwork, dangling = nametype(tn)[])
    names = Indices{nametype(tn)}(union(eachlinkname(tn), dangling))
    return normnetwork(tn, narrow_map(f, names))
end
function normnetwork(tn::ITensorNetwork, dangling = nametype(tn)[])
    return normnetwork(derive_braname, tn, dangling)
end
