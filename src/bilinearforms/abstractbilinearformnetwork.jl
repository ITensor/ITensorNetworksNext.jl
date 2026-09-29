using DataGraphs: DataGraphs, get_vertex_data, is_vertex_assigned
using Dictionaries: Dictionaries, Dictionary, isinsertable, issettable
using Graphs: Graphs, edges, vertices
using ITensorBase: ITensorBase, conj, inds, name, rename
using NamedGraphs: NamedGraphs, decoded_vertex, encoded_graph, encoded_vertex

"""
    abstract type AbstractBilinearFormNetwork{T, V, I} <: AbstractITensorNetwork{T, V}

Supertype of the lazy multi-layer networks built from a ket layer of type
`ITensorNetwork{T, V, I}` and a ket→bra index name mapping.

A subtype supplies its own graph structure, implements [`braname`](@ref), and returns an
[`AbstractGramian`](@ref) from `getindex`; [`kettensor`](@ref), [`bratensor`](@ref) and,
where the subtype has an operator layer, [`operatortensor`](@ref) read a vertex's layers from
that Gramian. The layers as whole networks are returned by [`ketnetwork`](@ref),
[`branetwork`](@ref) and [`operatornetwork`](@ref).
"""
abstract type AbstractBilinearFormNetwork{T, V, I} <: AbstractITensorNetwork{T, V} end

"""
    abstract type AbstractGramian

The layers of an `AbstractBilinearFormNetwork` at one vertex. A subtype implements
[`kettensor`](@ref), [`braname`](@ref), `layertensors` and `layerinds`; the bra tensor is built
from the ket tensor and the name map each time it is requested. Its `inds`, `names` and `axes`
are the indices of its layers that no other layer shares, those the layer product leaves open.
"""
abstract type AbstractGramian end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(bn::AbstractBilinearFormNetwork) = edges(ketnetwork(bn))
Graphs.vertices(bn::AbstractBilinearFormNetwork) = vertices(ketnetwork(bn))

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.encoded_vertex(bn::AbstractBilinearFormNetwork, vertex)
    return encoded_vertex(ketnetwork(bn), vertex)
end
function NamedGraphs.decoded_vertex(bn::AbstractBilinearFormNetwork, code::Integer)
    return decoded_vertex(ketnetwork(bn), code)
end
NamedGraphs.encoded_graph(bn::AbstractBilinearFormNetwork) = encoded_graph(ketnetwork(bn))

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.is_vertex_assigned(bn::AbstractBilinearFormNetwork, vertex)
    return isassigned(ketnetwork(bn), vertex)
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::AbstractBilinearFormNetwork) = false
Dictionaries.isinsertable(::AbstractBilinearFormNetwork) = false

# ====================================== interface ======================================= #

"""
    braname(bn::AbstractBilinearFormNetwork, name)
    braname(g::AbstractGramian, name)

The bra-layer index name corresponding to the ket-layer index name `name`. The `AbstractGramian`
form maps a name absent from its name map to itself, without checking it belongs to the network.
"""
function braname end
function braname(bn::AbstractBilinearFormNetwork, name)
    if !has_dimname(ketnetwork(bn), name)
        error("index name $name not found underlying tensor network.")
    end
    # A name absent from the map has no separate bra copy and maps to itself: a site index of a
    # norm network, or a site index a quadratic form's operator does not act on.
    return get(branamemap(bn), name, name)
end
braname(g::AbstractGramian, name) = get(branamemap(g), name, name)

"""
    branamemap(bn::AbstractBilinearFormNetwork)
    branamemap(g::AbstractGramian)

The ket→bra name map, holding a bra name for each ket index name that has a separate bra copy.
"""
function branamemap end

# A link name, or a name in `acted`, gets its bra name from `map`; every other name has none.
function select_branames(ket::ITensorNetwork{T, V, I}, map, acted) where {T, V, I}
    braname = Dictionary{I, I}()
    for (name, vertices) in pairs(ket.dimname_vertices)
        if length(vertices) == 2 || name in acted
            insert!(braname, name, map[name])
        end
    end
    return braname
end

"""
    kettensor(g::AbstractGramian)

The ket-layer tensor of the Gramian `g`.
"""
function kettensor end

"""
    operatortensor(g::AbstractGramian)

The operator-layer tensor of the Gramian `g`, with its index names renamed so that its input
legs meet the ket layer and its output legs meet the bra layer.
"""
function operatortensor end

conj_bratensor(g::AbstractGramian) = rename(n -> braname(g, n), kettensor(g))

"""
    bratensor(g::AbstractGramian)

The bra-layer tensor of the Gramian `g`.
"""
bratensor(g::AbstractGramian) = conj(conj_bratensor(g))

# Read from `conj_bratensor`, which only renames, so the tensor data is not conjugated.
brainds(g::AbstractGramian) = conj.(inds(conj_bratensor(g)))

function ITensorBase.inds(g::AbstractGramian)
    layer_inds = reduce(vcat, collect.(layerinds(g)))
    layer_names = name.(layer_inds)
    return [i for i in layer_inds if count(==(name(i)), layer_names) == 1]
end
ITensorBase.names(g::AbstractGramian) = name.(inds(g))
Base.axes(g::AbstractGramian) = Tuple(inds(g))

"""
    ketnetwork(bn::AbstractBilinearFormNetwork)

The ket-layer network of `bn`.
"""
function ketnetwork end

"""
    operatornetwork(bn::AbstractBilinearFormNetwork)

The operator-layer network of `bn`, for a subtype that has an operator layer.
"""
function operatornetwork end

"""
    branetwork(bn::AbstractBilinearFormNetwork)

The bra-layer network of `bn`. Unless a subtype stores its bra layer as a network, this is a
`BraView`, whose tensors are built by [`bratensor`](@ref) when accessed.
"""
branetwork(bn::AbstractBilinearFormNetwork) = BraView(bn)

"""
    struct BraView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <: AbstractITensorNetwork{T, V}

The bra layer of the bilinear-form network `parent(view)`, with each vertex tensor built by
[`bratensor`](@ref) when accessed. Its graph structure and mutability are those of the parent.
"""
struct BraView{T, V, I, P <: AbstractBilinearFormNetwork{T, V, I}} <:
    AbstractITensorNetwork{T, V}
    parent::P
    function BraView(parent::AbstractBilinearFormNetwork{T, V, I}) where {T, V, I}
        return new{T, V, I, typeof(parent)}(parent)
    end
end

Base.parent(nnv::BraView) = nnv.parent

# ==================================== DataGraphs.jl ===================================== #

DataGraphs.get_vertex_data(nnv::BraView, vertex) = bratensor(parent(nnv)[vertex])
function DataGraphs.is_vertex_assigned(nnv::BraView, vertex)
    return is_vertex_assigned(parent(nnv), vertex)
end

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

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(nnv::BraView) = issettable(parent(nnv))
Dictionaries.isinsertable(nnv::BraView) = isinsertable(parent(nnv))
