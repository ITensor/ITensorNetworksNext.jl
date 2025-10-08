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
using TermInterface: TermInterface, arguments, iscall, maketerm, operation, sorted_arguments

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
function Base.hash(m::Mul, h::UInt64)
    h = hash(:Mul, h)
    for arg in arguments(m)
        h = hash(arg, h)
    end
    return h
end
function map_arguments(f, m::Mul)
    return Mul(map(f, arguments(m)))
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

function getindex_lazy(a::AbstractArray, I...)
    u = unwrap(a)
    if !iscall(u)
        return u[I...]
    else
        return error("Indexing into expression not supported.")
    end
end
function Base.getindex(a::LazyNamedDimsArray, I::Int...)
    return getindex_lazy(a, I...)
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

# Defined to avoid type piracy.
# TODO: Define a proper hash function
# in NamedDimsArrays.jl, maybe one that is
# independent of the order of dimensions.
function _hash(a::NamedDimsArray, h::UInt64)
    h = hash(:NamedDimsArray, h)
    h = hash(dename(a), h)
    for i in inds(a)
        h = hash(i, h)
    end
    return h
end
function _hash(x, h::UInt64)
    return hash(x, h)
end
function Base.hash(a::LazyNamedDimsArray, h::UInt64)
    h = hash(:LazyNamedDimsArray, h)
    # Use `_hash`, which defines a custom hash for NamedDimsArray.
    return _hash(unwrap(a), h)
end

generic_map(f, v) = map(f, v)
generic_map(f, v::AbstractDict) = Dict(eachindex(v) .=> map(f, values(v)))
generic_map(f, v::AbstractSet) = Set([f(x) for x in v])
function map_arguments(f, a::LazyNamedDimsArray)
    u = unwrap(a)
    if !iscall(u)
        return error("No arguments to map.")
    elseif ismul(u)
        return LazyNamedDimsArray(map_arguments(f, u))
    else
        return error("Variant not supported.")
    end
end
function substitute(a::LazyNamedDimsArray, substitutions::AbstractDict)
    haskey(substitutions, a) && return substitutions[a]
    !iscall(a) && return a
    return map_arguments(arg -> substitute(arg, substitutions), a)
end
function substitute(a::LazyNamedDimsArray, substitutions)
    return substitute(a, Dict(substitutions))
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

struct SymbolicArray{T, N, Name, Axes <: NTuple{N, AbstractUnitRange{<:Integer}}} <: AbstractArray{T, N}
    name::Name
    axes::Axes
    function SymbolicArray{T}(name, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}}) where {T}
        N = length(ax)
        return new{T, N, typeof(name), typeof(ax)}(name, ax)
    end
end
function SymbolicArray(name, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    return SymbolicArray{Any}(name, ax)
end
function SymbolicArray{T}(name, ax::AbstractUnitRange...) where {T}
    return SymbolicArray{T}(name, ax)
end
function SymbolicArray(name, ax::AbstractUnitRange...)
    return SymbolicArray{Any}(name, ax)
end
symname(a::SymbolicArray) = getfield(a, :name)
Base.axes(a::SymbolicArray) = getfield(a, :axes)
Base.size(a::SymbolicArray) = length.(axes(a))
function Base.:(==)(a::SymbolicArray, b::SymbolicArray)
    return symname(a) == symname(b) && axes(a) == axes(b)
end
function Base.hash(a::SymbolicArray, h::UInt64)
    h = hash(:SymbolicArray, h)
    h = hash(symname(a), h)
    return hash(size(a), h)
end
function Base.getindex(a::SymbolicArray, I...)
    return error("Indexing into SymbolicArray not supported.")
end
function Base.setindex!(a::SymbolicArray, value, I...)
    return error("Indexing into SymbolicArray not supported.")
end
function Base.show(io::IO, mime::MIME"text/plain", a::SymbolicArray)
    Base.summary(io, a)
    println(io, ":")
    print(io, repr(symname(a)))
    return nothing
end
function Base.show(io::IO, a::SymbolicArray)
    print(io, "SymbolicArray(", symname(a), ", ", size(a), ")")
    return nothing
end
using AbstractTrees: AbstractTrees
function AbstractTrees.printnode(io::IO, a::SymbolicArray)
    print(io, repr(symname(a)))
    return nothing
end
const SymbolicNamedDimsArray{T, N, Parent <: SymbolicArray{T, N}, DimNames} =
    NamedDimsArray{T, N, Parent, DimNames}
function symnameddims(name)
    return lazy(NamedDimsArray(SymbolicArray(name), ()))
end
function printnode(io::IO, a::SymbolicNamedDimsArray)
    print(io, symname(dename(a)))
    if ndims(a) > 0
        print(io, "[", join(dimnames(a), ","), "]")
    end
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

end
