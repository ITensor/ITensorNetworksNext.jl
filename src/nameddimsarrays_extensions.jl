using NamedDimsArrays: AbstractNamedDimsArray
using VectorInterface: VectorInterface as VI

# Temporary `VectorInterface` methods for named arrays.
#
# `KrylovKit.eigsolve` drives its Krylov vectors through `VectorInterface`. The generic
# `AbstractArray` fallbacks broadcast in a way that fails on a named array (e.g.
# `zerovector(x, S)`), so we provide name-aware methods here. This is type piracy on
# `AbstractNamedDimsArray` and is intended to move into `NamedDimsArrays`.
VI.scalartype(::Type{<:AbstractNamedDimsArray{T}}) where {T} = T
function VI.zerovector(x::AbstractNamedDimsArray, ::Type{S}) where {S <: Number}
    return fill!(similar(x, S), zero(S))
end
VI.scale(x::AbstractNamedDimsArray, α::Number) = x * α
VI.scale!!(x::AbstractNamedDimsArray, α::Number) = x * α
VI.scale!!(::AbstractNamedDimsArray, x::AbstractNamedDimsArray, α::Number) = x * α
function VI.add!!(
        y::AbstractNamedDimsArray, x::AbstractNamedDimsArray, α::Number, β::Number
    )
    return x * α + y * β
end
VI.inner(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray) = (conj(x) * y)[]
