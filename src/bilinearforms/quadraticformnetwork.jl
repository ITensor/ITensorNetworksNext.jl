using Dictionaries: Dictionary
using ITensorBase: LazyNamedTensor, dimnames, inputnames, lazy, outputnames,
    replacedimnames, state, uniquename
using ITensorNetworksNext

"""
    struct QuadraticFormNetwork{T, V, I, O} <: AbstractBilinearFormNetwork{T, V, I}

Lazy wrapper representing the quadratic form `⟨tn|op|tn⟩` of `tn::ITensorNetwork{T, V, I}`
sandwiched around the operator layer `op::ITensorNetworkOperator`, together with a per-index
ket→bra name mapping that, for each index in the ket layer, defines the name of the
corresponding index in the bra layer.

The operator layer must be defined on every vertex of `tn`. Its input names are ket index
names, and each paired output name is renamed to the bra layer, so the operator's remaining
index names must be distinct from every index name of `tn`.
"""
struct QuadraticFormNetwork{T, V, I, O <: ITensorNetworkOperator} <:
    AbstractBilinearFormNetwork{T, V, I}
    ket::ITensorNetwork{T, V, I}
    operator::O
    braname::Dictionary{I, I}
    function QuadraticFormNetwork(
            ket::ITensorNetwork{T, V, I},
            operator::ITensorNetworkOperator,
            map::Dictionary{I, I}
        ) where {T, V, I}
        if !issetequal(vertices(operator), vertices(ket))
            error("the operator layer must be defined on every vertex of the ket layer.")
        end
        acted = Set{I}(inputnames(operator))
        braname = Dictionary{I, I}()
        for (name, vertices) in pairs(ket.dimname_vertices)
            if length(vertices) == 2 || name in acted
                insert!(braname, name, map[name])
            end
        end
        return new{T, V, I, typeof(operator)}(ket, operator, braname)
    end
end

Base.eltype(::Type{<:QuadraticFormNetwork{T, V, I}}) where {T, V, I} = LazyNamedTensor{I, T}

function QuadraticFormNetwork(ket::ITensorNetwork, operator::ITensorNetworkOperator)
    return QuadraticFormNetwork(ket, operator, map(uniquename, keys(ket.dimname_vertices)))
end

# ====================================== Graphs.jl ======================================= #

Graphs.edges(qf::QuadraticFormNetwork) = edges(qf.ket)
Graphs.vertices(qf::QuadraticFormNetwork) = vertices(qf.ket)

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.encoded_vertex(qf::QuadraticFormNetwork, vertex)
    return encoded_vertex(qf.ket, vertex)
end
function NamedGraphs.decoded_vertex(qf::QuadraticFormNetwork, code::Integer)
    return decoded_vertex(qf.ket, code)
end
NamedGraphs.encoded_graph(qf::QuadraticFormNetwork) = encoded_graph(qf.ket)

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(qf::QuadraticFormNetwork, vertex)
    A = kettensor(qf, vertex)
    O = operatortensor(qf, vertex)
    B = conj_bratensor(qf, vertex)
    # TODO: implement and use a lazy `conj` via `LazyNamedDimsArrays` here?
    return lazy(A) * lazy(O) * lazy(conj(B))
end

function DataGraphs.is_vertex_assigned(qf::QuadraticFormNetwork, vertex)
    return isassigned(qf.ket, vertex) && isassigned(qf.operator, vertex)
end

# ====================================== interface ======================================= #

function braname(qf::QuadraticFormNetwork, name)
    if !has_dimname(qf.ket, name)
        error("index name $name not found underlying tensor network.")
    end
    # The indices not stored in `qf.braname` are the dangling ket indices the operator does
    # not act on, which get mapped to themselves.
    return get(qf.braname, name, name)
end

kettensor(qf::QuadraticFormNetwork, vertex) = qf.ket[vertex]

# Each output name is renamed to the bra name of the input name it is paired with, so the
# operator's output legs meet the bra layer and its input legs meet the ket layer. The pairing
# is read from the operator network rather than from `qf.operator[vertex]`, whose wrapper drops
# a pair whose input sits on another vertex.
function operatortensor(qf::QuadraticFormNetwork, vertex)
    tensor = state(qf.operator)[vertex]
    names = dimnames(tensor)
    replacements = [
        output => braname(qf, input) for
            (output, input) in zip(outputnames(qf.operator), inputnames(qf.operator))
            if output in names
    ]
    return replacedimnames(tensor, replacements...)
end

"""
    quadraticformnetwork(tn::ITensorNetwork, op::ITensorNetworkOperator, [braname])

Build the triple-layer network `⟨tn|op|tn⟩`, represented lazily as a
`QuadraticFormNetwork` object. The optional third argument `braname` should implement
`braname[ketdimname] = bradimname` for every link dimension name `ketdimname` in `tn` and
every dimension name of `tn` the operator acts on. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
function quadraticformnetwork(tn::ITensorNetwork, op::ITensorNetworkOperator)
    return QuadraticFormNetwork(tn, op)
end
function quadraticformnetwork(tn::ITensorNetwork, op::ITensorNetworkOperator, braname)
    return QuadraticFormNetwork(tn, op, braname)
end
