using Dictionaries: Dictionary, Indices, getindices
using ITensorBase: inputnames, outputnames, rename, state, uniquename
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
        if !issubset(inputnames(operator), keys(ket.dimname_vertices))
            error("every operator input name must be an index name of the ket layer.")
        end
        braname = getindices(map, Indices{I}(union(linknames(ket), inputnames(operator))))
        return new{T, V, I, typeof(operator)}(ket, operator, braname)
    end
end

function Base.eltype(::Type{<:QuadraticFormNetwork})
    return error(
        "`eltype` of a `QuadraticFormNetwork` is not defined, since the triple-layer tensor at a vertex has no representation of its own. Use `kettensor`, `operatortensor` and `bratensor` to reach the individual layers."
    )
end

function QuadraticFormNetwork(ket::ITensorNetwork, operator::ITensorNetworkOperator)
    return QuadraticFormNetwork(ket, operator, map(uniquename, keys(ket.dimname_vertices)))
end

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(qf::QuadraticFormNetwork, vertex)
    return error(
        "Indexing a `QuadraticFormNetwork` is not defined, since the triple-layer tensor at a vertex has no representation of its own. Use `kettensor`, `operatortensor` and `bratensor` to reach the individual layers."
    )
end

function DataGraphs.is_vertex_assigned(qf::QuadraticFormNetwork, vertex)
    return isassigned(ketnetwork(qf), vertex) && isassigned(operatornetwork(qf), vertex)
end

# ====================================== interface ======================================= #

ketnetwork(qf::QuadraticFormNetwork) = qf.ket
branamemap(qf::QuadraticFormNetwork) = qf.braname
operatornetwork(qf::QuadraticFormNetwork) = qf.operator

# Each output name is renamed to the bra name of the input it is paired with, so the output legs
# meet the bra layer and the input legs meet the ket layer.
function operatortensor(qf::QuadraticFormNetwork, vertex)
    op = qf.operator[vertex]
    replacements = [
        output => braname(qf, input) for
            (output, input) in zip(outputnames(op), inputnames(op))
    ]
    return rename(state(op), replacements...)
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
