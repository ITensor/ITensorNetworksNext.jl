module LazyNamedDimsArrays

using WrappedUnions: @wrapped, unwrap
using NamedDimsArrays:
    NamedDimsArrays,
    AbstractNamedDimsArray,
    AbstractNamedDimsArrayStyle,
    dename,
    inds

struct Prod{A}
    factors::Vector{A}
end

@wrapped struct LazyNamedDimsArray{
        T, A <: AbstractNamedDimsArray{T},
    } <: AbstractNamedDimsArray{T, Any}
    union::Union{A, Prod{LazyNamedDimsArray{T, A}}}
end

function NamedDimsArrays.inds(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return inds(unwrap(a))
    elseif unwrap(a) isa Prod
        return mapreduce(inds, symdiff, unwrap(a).factors)
    else
        return error("Variant not supported.")
    end
end
function NamedDimsArrays.dename(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return dename(unwrap(a))
    elseif unwrap(a) isa Prod
        return dename(materialize(a), inds(a))
    else
        return error("Variant not supported.")
    end
end

using Base.Broadcast: materialize
function Base.Broadcast.materialize(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return unwrap(a)
    elseif unwrap(a) isa Prod
        return prod(materialize, unwrap(a).factors)
    else
        return error("Variant not supported.")
    end
end
Base.copy(a::LazyNamedDimsArray) = materialize(a)

function Base.:*(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return LazyNamedDimsArray(Prod([lazy(unwrap(a))]))
    elseif unwrap(a) isa Prod
        return a
    else
        return error("Variant not supported.")
    end
end

function Base.:*(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray)
    # Nested by default.
    return LazyNamedDimsArray(Prod([a1, a2]))
end
function Base.:+(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray)
    return error("Not implemented.")
end
function Base.:-(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray)
    return error("Not implemented.")
end
function Base.:*(c::Number, a::LazyNamedDimsArray)
    return error("Not implemented.")
end
function Base.:*(a::LazyNamedDimsArray, c::Number)
    return error("Not implemented.")
end
function Base.:/(a::LazyNamedDimsArray, c::Number)
    return error("Not implemented.")
end
function Base.:-(a::LazyNamedDimsArray)
    return error("Not implemented.")
end

function LazyNamedDimsArray(a::AbstractNamedDimsArray)
    return LazyNamedDimsArray{eltype(a), typeof(a)}(a)
end
function LazyNamedDimsArray(a::Prod{LazyNamedDimsArray{T, A}}) where {T, A}
    return LazyNamedDimsArray{T, A}(a)
end
function lazy(a::AbstractNamedDimsArray)
    return LazyNamedDimsArray(a)
end

# Broadcasting
struct LazyNamedDimsArrayStyle <: AbstractNamedDimsArrayStyle{Any} end
function Base.BroadcastStyle(::Type{<:LazyNamedDimsArray})
    return LazyNamedDimsArrayStyle()
end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, f, as...)
    return error("Arbitrary broadcasting not supported for LazyNamedDimsArray.")
end
# Linear operations.
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(+), a1, a2)
    return a1 + a2
end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(-), a1, a2)
    return a1 - a2
end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), c::Number, a)
    return c * a
end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), a, c::Number)
    return a * c
end
# Fix ambiguity error.
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), a::Number, b::Number)
    return a * b
end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(/), a, c::Number)
    return a / c
end
function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(-), a)
    return -a
end

using TermInterface: TermInterface
# arguments, arity, children, head, iscall, operation
function TermInterface.arguments(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return error("No arguments.")
    elseif unwrap(a) isa Prod
        unwrap(a).factors
    else
        return error("Variant not supported.")
    end
end
function TermInterface.children(a::LazyNamedDimsArray)
    return TermInterface.arguments(a)
end
function TermInterface.head(a::LazyNamedDimsArray)
    return TermInterface.operation(a)
end
function TermInterface.iscall(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return false
    elseif unwrap(a) isa Prod
        return true
    else
        return false
    end
end
function TermInterface.isexpr(a::LazyNamedDimsArray)
    return TermInterface.iscall(a)
end
function TermInterface.maketerm(::Type{LazyNamedDimsArray}, head, args, metadata)
    if head ≡ prod
        return LazyNamedDimsArray(Prod(args))
    else
        return error("Only product terms supported right now.")
    end
end
function TermInterface.operation(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return error("No operation.")
    elseif unwrap(a) isa Prod
        prod
    else
        return error("Variant not supported.")
    end
end
function TermInterface.sorted_arguments(a::LazyNamedDimsArray)
    if unwrap(a) isa AbstractNamedDimsArray
        return error("No arguments.")
    elseif unwrap(a) isa Prod
        return TermInterface.arguments(a)
    else
        return error("Variant not supported.")
    end
end

end
