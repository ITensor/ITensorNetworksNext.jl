using Dictionaries: Dictionary, Indices, dictionary
using ITensorBase: ITensorBase, conj, nametype, rename, similar_operator
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
        check_input(NormNetwork, ket, map)
        return new{T, V, I}(ket, map)
    end
end

function NormNetwork(
        f::Function, tn::ITensorNetwork, linknames = ITensorNetworksNext.linknames(tn)
    )
    return NormNetwork(tn, narrow_map(f, Indices(linknames)))
end
function NormNetwork(tn::ITensorNetwork, linknames = ITensorNetworksNext.linknames(tn))
    return NormNetwork(name -> derivedname(name, "bra"), tn, linknames)
end

function check_input(::Type{<:NormNetwork}, ket, braname)
    if !issubset(linknames(ket), keys(braname))
        error("every link name of the ket layer must have a bra name.")
    end
    if !allunique(braname)
        error("the bra names of a norm network must be distinct.")
    end
    for name in braname
        if has_dimname(ket, name)
            error("bra name $name is already an index name of the ket layer.")
        end
    end
    return nothing
end

function Base.eltype(::Type{<:NormNetwork})
    return error(
        "`eltype` of a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(nn::NormNetwork) = edges(ketnetwork(nn))
Graphs.vertices(nn::NormNetwork) = vertices(ketnetwork(nn))

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.encoded_vertex(nn::NormNetwork, vertex)
    return encoded_vertex(ketnetwork(nn), vertex)
end
function NamedGraphs.decoded_vertex(nn::NormNetwork, code::Integer)
    return decoded_vertex(ketnetwork(nn), code)
end
NamedGraphs.encoded_graph(nn::NormNetwork) = encoded_graph(ketnetwork(nn))

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(nn::NormNetwork, vertex)
    return error(
        "Indexing a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

function DataGraphs.is_vertex_assigned(nn::NormNetwork, vertex)
    return isassigned(ketnetwork(nn), vertex)
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::NormNetwork) = false
Dictionaries.isinsertable(::NormNetwork) = false

# ====================================== interface ======================================= #

# The name type of the indices, the same as the ket network's.
ITensorBase.nametype(::Type{<:NormNetwork{T, V, I}}) where {T, V, I} = I
ITensorBase.nametype(nn::NormNetwork) = nametype(typeof(nn))

"""
    ketnetwork(nn::NormNetwork)

The ket-layer network of `nn`.
"""
ketnetwork(nn::NormNetwork) = nn.ket

"""
    branetwork(nn::NormNetwork)

The bra-layer network of `nn`, a `BraView` whose tensors are built by [`bratensor`](@ref) when
accessed.
"""
branetwork(nn::NormNetwork) = BraView(nn)

"""
    branamemap(nn::NormNetwork)

The ket→bra name map, holding a bra name for each ket index name that has a separate bra copy.
"""
branamemap(nn::NormNetwork) = nn.braname

"""
    braname(nn::NormNetwork, name)

The bra-layer index name corresponding to the ket-layer index name `name`.
"""
function braname(nn::NormNetwork, name)
    if !has_dimname(ketnetwork(nn), name)
        error("index name $name not found underlying tensor network.")
    end
    # The names absent from the map are the site indices, which map to themselves.
    return get(branamemap(nn), name, name)
end

"""
    kettensor(nn::NormNetwork, vertex)

The ket-layer tensor at `vertex`.
"""
kettensor(nn::NormNetwork, vertex) = ketnetwork(nn)[vertex]

function conj_bratensor(nn::NormNetwork, vertex)
    return rename(n -> braname(nn, n), kettensor(nn, vertex))
end

"""
    bratensor(nn::NormNetwork, vertex)

The bra-layer tensor at `vertex`.
"""
bratensor(nn::NormNetwork, vertex) = conj(conj_bratensor(nn, vertex))

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
    normnetwork([f::Function], tn::ITensorNetwork, [linknames]) -> NormNetwork
    normnetwork(tn::ITensorNetwork, braname::Dictionary) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NormNetwork` object.
Each ket index name in `linknames` (by default `linknames(tn)`) gets the bra name `f(name)`,
where `f` defaults to `name -> derivedname(name, "bra")`. Alternatively pass the ket→bra name
map `braname` directly. Either way the names mapped must include every link name of `tn` and
may include others, and an error is thrown if a link name is missing, two bra names coincide
or a bra name is already an index name of `tn`.
"""
normnetwork(args...) = NormNetwork(args...)
