using Dictionaries: Dictionary, dictionary
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

function Base.eltype(::Type{<:NormNetwork})
    return error(
        "`eltype` of a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

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

function DataGraphs.get_vertex_data(nn::NormNetwork, vertex)
    return error(
        "Indexing a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
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

kettensor(nn::NormNetwork, vertex) = nn.ket[vertex]

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
    normnetwork(tn::ITensorNetwork, [braname]) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NomnNetwork` object.
The optional second argument `braname` should implement `braname[ketdimname] = bradimname` for
every link dimension name `ketdimname` in `tn`. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
normnetwork(tn::ITensorNetwork) = NormNetwork(tn)
normnetwork(tn::ITensorNetwork, braname) = NormNetwork(tn, braname)
