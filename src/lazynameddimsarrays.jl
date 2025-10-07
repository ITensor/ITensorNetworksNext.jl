module LazyNamedDimsArrays

using WrappedUnions: @wrapped, unwrap
using NamedDimsArrays:
    NamedDimsArrays,
    AbstractNamedDimsArray,
    AbstractNamedDimsArrayStyle,
    dename,
    inds
using TermInterface: TermInterface, arguments, iscall, maketerm, operation, sorted_arguments

struct Mul{A}
    arguments::Vector{A}
end
TermInterface.arguments(m::Mul) = getfield(m, :arguments)
TermInterface.children(m::Mul) = arguments(m)
TermInterface.head(m::Mul) = operation(m)
TermInterface.iscall(m::Mul) = true
TermInterface.isexpr(m::Mul) = iscall(m)
TermInterface.maketerm(::Type{Mul}, head::typeof(*), args, metadata) = Mul(args)
TermInterface.operation(m::Mul) = *
TermInterface.sorted_arguments(m::Mul) = arguments(m)
TermInterface.sorted_children(m::Mul) = sorted_arguments(a)

@wrapped struct LazyNamedDimsArray{
        T, A <: AbstractNamedDimsArray{T},
    } <: AbstractNamedDimsArray{T, Any}
    union::Union{A, Mul{LazyNamedDimsArray{T, A}}}
end

function NamedDimsArrays.inds(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return inds(u)
    elseif u isa Mul
        return mapreduce(inds, symdiff, arguments(u))
    else
        return error("Variant not supported.")
    end
end
function NamedDimsArrays.dename(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return dename(u)
    elseif u isa Mul
        return dename(materialize(a), inds(a))
    else
        return error("Variant not supported.")
    end
end

function TermInterface.arguments(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return error("No arguments.")
    elseif u isa Mul
        return arguments(u)
    else
        return error("Variant not supported.")
    end
end
function TermInterface.children(a::LazyNamedDimsArray)
    return arguments(a)
end
function TermInterface.head(a::LazyNamedDimsArray)
    return operation(a)
end
function TermInterface.iscall(a::LazyNamedDimsArray)
    return iscall(unwrap(a))
end
function TermInterface.isexpr(a::LazyNamedDimsArray)
    return iscall(a)
end
function TermInterface.maketerm(::Type{LazyNamedDimsArray}, head, args, metadata)
    if head ≡ *
        return LazyNamedDimsArray(maketerm(Mul, head, args, metadata))
    else
        return error("Only product terms supported right now.")
    end
end
function TermInterface.operation(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return error("No operation.")
    elseif u isa Mul
        return operation(u)
    else
        return error("Variant not supported.")
    end
end
function TermInterface.sorted_arguments(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return error("No arguments.")
    elseif u isa Mul
        return sorted_arguments(u)
    else
        return error("Variant not supported.")
    end
end
function TermInterface.sorted_children(a::LazyNamedDimsArray)
    return sorted_arguments(a)
end

using Base.Broadcast: materialize
function Base.Broadcast.materialize(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return u
    elseif u isa Mul
        return mapfoldl(materialize, operation(u), arguments(u))
    else
        return error("Variant not supported.")
    end
end
Base.copy(a::LazyNamedDimsArray) = materialize(a)

function Base.:*(a::LazyNamedDimsArray)
    u = unwrap(a)
    if u isa AbstractNamedDimsArray
        return LazyNamedDimsArray(Mul([lazy(u)]))
    elseif u isa Mul
        return a
    else
        return error("Variant not supported.")
    end
end

function Base.:*(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray)
    # Nested by default.
    return LazyNamedDimsArray(Mul([a1, a2]))
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
function LazyNamedDimsArray(a::Mul{LazyNamedDimsArray{T, A}}) where {T, A}
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

end
