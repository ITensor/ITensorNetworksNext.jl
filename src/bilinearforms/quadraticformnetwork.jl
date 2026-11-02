using Dictionaries: Dictionary, Indices
using ITensorBase: inputnames, outputnames, rename, state
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
        check_input(QuadraticFormNetwork, ket, operator, map)
        return new{T, V, I, typeof(operator)}(ket, operator, map)
    end
end

function QuadraticFormNetwork(
        f::Function,
        ket::ITensorNetwork,
        operator::ITensorNetworkOperator,
        linknames = ITensorNetworksNext.linknames(ket)
    )
    names = union(linknames, inputnames(operator))
    return QuadraticFormNetwork(ket, operator, narrow_map(f, Indices(names)))
end
function QuadraticFormNetwork(
        ket::ITensorNetwork,
        operator::ITensorNetworkOperator,
        linknames = ITensorNetworksNext.linknames(ket)
    )
    return QuadraticFormNetwork(name -> derivedname(name, "bra"), ket, operator, linknames)
end

function check_input(::Type{<:QuadraticFormNetwork}, ket, operator, braname)
    if !issetequal(vertices(operator), vertices(ket))
        error("the operator layer must be defined on every vertex of the ket layer.")
    end
    inputs = inputnames(operator)
    for input in inputs
        if !has_dimname(ket, input)
            error("operator input name $input is not an index name of the ket layer.")
        end
        if dimnamevertices(ket, input) != dimnamevertices(state(operator), input)
            error("operator input name $input must be a ket site index on the same vertex.")
        end
    end
    for vertex in vertices(operator), name in names(state(operator), vertex)
        if name ∉ inputs && has_dimname(ket, name)
            error("operator index name $name is not an input but is a ket index name.")
        end
    end
    if !issubset(inputs, keys(braname))
        error("every operator input name must have a bra name.")
    end
    return check_branamemap(ket, braname, state(operator))
end

function Base.eltype(::Type{<:QuadraticFormNetwork})
    return error(
        "`eltype` of a `QuadraticFormNetwork` is not defined, since the triple-layer tensor \
        at a vertex has no representation of its own. Use `kettensor`, `operatortensor` and \
        `bratensor` to reach the individual layers."
    )
end

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(qf::QuadraticFormNetwork, vertex)
    return error(
        "Indexing a `QuadraticFormNetwork` is not defined, since the triple-layer tensor at \
        a vertex has no representation of its own. Use `kettensor`, `operatortensor` and \
        `bratensor` to reach the individual layers."
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
    quadraticformnetwork([f::Function], tn::ITensorNetwork, op::ITensorNetworkOperator, [linknames])
    quadraticformnetwork(tn::ITensorNetwork, op::ITensorNetworkOperator, braname::Dictionary)

Build the triple-layer network `⟨tn|op|tn⟩`, represented lazily as a `QuadraticFormNetwork`
object. Each ket index name in `linknames` (by default `linknames(tn)`) and each input name of
`op` gets the bra name `f(name)`, where `f` defaults to `name -> derivedname(name, "bra")`.
Alternatively pass the ket→bra name map `braname` directly. Either way the names mapped must
include every link name of `tn` and every input name of `op` and may include others, and an
error is thrown if one is missing, two bra names coincide or a bra name is already an index
name of `tn` or `op`.
"""
quadraticformnetwork(args...) = QuadraticFormNetwork(args...)
