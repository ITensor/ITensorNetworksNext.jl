module LazyNamedDimsArrays

using AbstractTrees: AbstractTrees
using WrappedUnions: @wrapped, unwrap
using NamedDimsArrays:
    NamedDimsArrays,
    AbstractNamedDimsArray,
    AbstractNamedDimsArrayStyle,
    NamedDimsArray,
    dename,
    dimnames,
    inds
using ..SymbolicArrays: SymbolicArrays, SymbolicArray
using TermInterface: TermInterface, arguments, iscall, maketerm, operation, sorted_arguments

const SymbolicNamedDimsArray{T, N, Parent <: SymbolicArray{T, N}, DimNames} =
    NamedDimsArray{T, N, Parent, DimNames}
function symnameddims(name)
    return lazy(NamedDimsArray(SymbolicArray(name), ()))
end
function printnode(io::IO, a::SymbolicNamedDimsArray)
    print(io, SymbolicArrays.name(dename(a)))
    print(io, "[", join(dimnames(a), ","), "]")
    return nothing
end
function Base.:(==)(a::SymbolicNamedDimsArray, b::SymbolicNamedDimsArray)
    return issetequal(inds(a), inds(b)) && dename(a) == dename(b)
end
function Base.:*(a::SymbolicNamedDimsArray, b::SymbolicNamedDimsArray)
    return lazy(a) * lazy(b)
end
function Base.:*(a::SymbolicNamedDimsArray, b::LazyNamedDimsArray)
    return lazy(a) * b
end
function Base.:*(a::LazyNamedDimsArray, b::SymbolicNamedDimsArray)
    return a * lazy(b)
end

# Custom version of `AbstractTrees.printnode` to
# avoid type piracy when overloading on `AbstractNamedDimsArray`.
printnode(io::IO, x) = AbstractTrees.printnode(io, x)
function printnode(io::IO, a::AbstractNamedDimsArray)
    show(io, collect(dimnames(a)))
    return nothing
end

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
ismul(x) = false
ismul(m::Mul) = true
function Base.show(io::IO, m::Mul)
    args = map(arg -> sprint(printnode, arg), arguments(m))
    print(io, "(", join(args, " $(operation(m)) "), ")")
    return nothing
end

@wrapped struct LazyNamedDimsArray{
        T, A <: AbstractNamedDimsArray{T},
    } <: AbstractNamedDimsArray{T, Any}
    union::Union{A, Mul{LazyNamedDimsArray{T, A}}}
end

function NamedDimsArrays.inds(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return inds(u)
    elseif ismul(u)
        return mapreduce(inds, symdiff, arguments(u))
    else
        return error("Variant not supported.")
    end
end
function NamedDimsArrays.dename(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return dename(u)
    else
        return error("Variant not supported.")
    end
end

function TermInterface.arguments(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return error("No arguments.")
    elseif ismul(u)
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
        return error("Only mul supported right now.")
    end
end
function TermInterface.operation(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return error("No operation.")
    elseif ismul(u)
        return operation(u)
    else
        return error("Variant not supported.")
    end
end
function TermInterface.sorted_arguments(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return error("No arguments.")
    elseif ismul(u)
        return sorted_arguments(u)
    else
        return error("Variant not supported.")
    end
end
function TermInterface.sorted_children(a::LazyNamedDimsArray)
    return sorted_arguments(a)
end
ismul(a::LazyNamedDimsArray) = ismul(unwrap(a))

function AbstractTrees.children(a::LazyNamedDimsArray)
    if !iscall(a)
        return ()
    else
        return arguments(a)
    end
end
function AbstractTrees.nodevalue(a::LazyNamedDimsArray)
    if !iscall(a)
        return unwrap(a)
    else
        return operation(a)
    end
end

using Base.Broadcast: materialize
function Base.Broadcast.materialize(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return u
    elseif ismul(u)
        return mapfoldl(materialize, operation(u), arguments(u))
    else
        return error("Variant not supported.")
    end
end
Base.copy(a::LazyNamedDimsArray) = materialize(a)

function Base.:(==)(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray)
    u1, u2 = unwrap.((a1, a2))
    if !iscall(u1) && !iscall(u2)
        return u1 == u2
    elseif ismul(u1) && ismul(u2)
        return arguments(u1) == arguments(u2)
    else
        return false
    end
end

function printnode(io::IO, a::LazyNamedDimsArray)
    return printnode(io, unwrap(a))
end
function AbstractTrees.printnode(io::IO, a::LazyNamedDimsArray)
    return printnode(io, a)
end
function Base.show(io::IO, a::LazyNamedDimsArray)
    if !iscall(a)
        return show(io, unwrap(a))
    else
        return printnode(io, a)
    end
end
function Base.show(io::IO, mime::MIME"text/plain", a::LazyNamedDimsArray)
    if !iscall(a)
        @invoke show(io, mime, a::AbstractNamedDimsArray)
        return nothing
    else
        show(io, a)
        return nothing
    end
end

function Base.:*(a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return LazyNamedDimsArray(Mul([lazy(u)]))
    elseif ismul(u)
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
