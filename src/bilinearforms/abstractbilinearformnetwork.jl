using Dictionaries: Dictionaries
using ITensorBase: conj, name, rename, setname

"""
    abstract type AbstractBilinearFormNetwork{T, V, I} <: AbstractITensorNetwork{T, V}

Supertype of the lazy multi-layer networks built from a ket layer of type
`ITensorNetwork{T, V, I}` and a ket→bra index name mapping.

A subtype supplies its own graph structure and implements [`braname`](@ref) together with one
accessor per layer: [`kettensor`](@ref), [`bratensor`](@ref) and, where the subtype has an
operator layer, [`operatortensor`](@ref). `bratensor` has a default built from `kettensor`
and `braname`. The layers as whole networks are returned by [`ketnetwork`](@ref),
[`branetwork`](@ref) and [`operatornetwork`](@ref).
"""
abstract type AbstractBilinearFormNetwork{T, V, I} <: AbstractITensorNetwork{T, V} end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::AbstractBilinearFormNetwork) = false
Dictionaries.isinsertable(::AbstractBilinearFormNetwork) = false

# ====================================== interface ======================================= #

"""
    braname(bn::AbstractBilinearFormNetwork, name)

The bra-layer index name corresponding to the ket-layer index name `name`.
"""
function braname end

"""
    kettensor(bn::AbstractBilinearFormNetwork, vertex)

The ket-layer tensor at `vertex`.
"""
function kettensor end

"""
    operatortensor(bn::AbstractBilinearFormNetwork, vertex)

The operator-layer tensor at `vertex`, with its index names renamed so that its input legs
meet the ket layer and its output legs meet the bra layer.
"""
function operatortensor end

function conj_bratensor(bn::AbstractBilinearFormNetwork, vertex)
    return rename(n -> braname(bn, n), kettensor(bn, vertex))
end

"""
    bratensor(bn::AbstractBilinearFormNetwork, vertex)

The bra-layer tensor at `vertex`.
"""
bratensor(bn::AbstractBilinearFormNetwork, vertex) = conj(conj_bratensor(bn, vertex))

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

indmap(bn::AbstractBilinearFormNetwork, ind) = setname(conj(ind), braname(bn, name(ind)))
