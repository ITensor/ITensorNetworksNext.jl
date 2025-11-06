module LazyNamedDimsArrays

include("baseextensions.jl")
include("nameddimsarraysextensions.jl")
include("symbolicarray.jl")
include("applied.jl")
include("lazyinterface.jl")
include("lazybroadcast.jl")
include("lazynameddimsarray.jl")
include("symbolicnameddimsarray.jl")

## using AbstractTrees: AbstractTrees
## using WrappedUnions: @wrapped, unwrap
## using NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray, AbstractNamedDimsArrayStyle,
##     NamedDimsArray, dename, dimnames, inds
## using TermInterface: TermInterface, arguments, iscall, maketerm, operation, sorted_arguments
## using TypeParameterAccessors: unspecify_type_parameters

## # Defined to avoid type piracy.
## # TODO: Define a proper hash function
## # in NamedDimsArrays.jl, maybe one that is
## # independent of the order of dimensions.
## function _hash(a::NamedDimsArray, h::UInt64)
##     h = hash(:NamedDimsArray, h)
##     h = hash(dename(a), h)
##     for i in inds(a)
##         h = hash(i, h)
##     end
##     return h
## end
## function _hash(x, h::UInt64)
##     return hash(x, h)
## end
##
## # Custom version of `AbstractTrees.printnode` to
## # avoid type piracy when overloading on `AbstractNamedDimsArray`.
## printnode_nameddims(io::IO, x) = AbstractTrees.printnode(io, x)
## function printnode_nameddims(io::IO, a::AbstractNamedDimsArray)
##     show(io, collect(dimnames(a)))
##     return nothing
## end

## # Generic lazy functionality.
## function maketerm_lazy(type::Type, head, args, metadata)
##     if head ≡ *
##         return type(maketerm(Mul, head, args, metadata))
##     else
##         return error("Only mul supported right now.")
##     end
## end
## function getindex_lazy(a::AbstractArray, I...)
##     u = unwrap(a)
##     if !iscall(u)
##         return u[I...]
##     else
##         return error("Indexing into expression not supported.")
##     end
## end
## function arguments_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return error("No arguments.")
##     elseif ismul(u)
##         return arguments(u)
##     else
##         return error("Variant not supported.")
##     end
## end
## function children_lazy(a)
##     return arguments(a)
## end
## function head_lazy(a)
##     return operation(a)
## end
## function iscall_lazy(a)
##     return iscall(unwrap(a))
## end
## function isexpr_lazy(a)
##     return iscall(a)
## end
## function operation_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return error("No operation.")
##     elseif ismul(u)
##         return operation(u)
##     else
##         return error("Variant not supported.")
##     end
## end
## function sorted_arguments_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return error("No arguments.")
##     elseif ismul(u)
##         return sorted_arguments(u)
##     else
##         return error("Variant not supported.")
##     end
## end
## function sorted_children_lazy(a)
##     return sorted_arguments(a)
## end
## ismul_lazy(a) = ismul(unwrap(a))
## function abstracttrees_children_lazy(a)
##     if !iscall(a)
##         return ()
##     else
##         return arguments(a)
##     end
## end
## function nodevalue_lazy(a)
##     if !iscall(a)
##         return unwrap(a)
##     else
##         return operation(a)
##     end
## end
## using Base.Broadcast: materialize
## function materialize_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return u
##     elseif ismul(u)
##         return mapfoldl(materialize, operation(u), arguments(u))
##     else
##         return error("Variant not supported.")
##     end
## end
## copy_lazy(a) = materialize(a)
## function equals_lazy(a1, a2)
##     u1, u2 = unwrap.((a1, a2))
##     if !iscall(u1) && !iscall(u2)
##         return u1 == u2
##     elseif ismul(u1) && ismul(u2)
##         return arguments(u1) == arguments(u2)
##     else
##         return false
##     end
## end
## function hash_lazy(a, h::UInt64)
##     h = hash(Symbol(unspecify_type_parameters(typeof(a))), h)
##     # Use `_hash`, which defines a custom hash for NamedDimsArray.
##     return _hash(unwrap(a), h)
## end
## function map_arguments_lazy(f, a)
##     u = unwrap(a)
##     if !iscall(u)
##         return error("No arguments to map.")
##     elseif ismul(u)
##         return lazy(map_arguments(f, u))
##     else
##         return error("Variant not supported.")
##     end
## end
## function substitute_lazy(a, substitutions::AbstractDict)
##     haskey(substitutions, a) && return substitutions[a]
##     !iscall(a) && return a
##     return map_arguments(arg -> substitute(arg, substitutions), a)
## end
## function substitute_lazy(a, substitutions)
##     return substitute(a, Dict(substitutions))
## end
## function printnode_lazy(io, a)
##     # Use `printnode_nameddims` to avoid type piracy,
##     # since it overloads on `AbstractNamedDimsArray`.
##     return printnode_nameddims(io, unwrap(a))
## end
## function show_lazy(io::IO, a)
##     if !iscall(a)
##         return show(io, unwrap(a))
##     else
##         return AbstractTrees.printnode(io, a)
##     end
## end
## function show_lazy(io::IO, mime::MIME"text/plain", a)
##     summary(io, a)
##     println(io, ":")
##     if !iscall(a)
##         show(io, mime, unwrap(a))
##         return nothing
##     else
##         show(io, a)
##         return nothing
##     end
## end
## add_lazy(a1, a2) = error("Not implemented.")
## sub_lazy(a) = error("Not implemented.")
## sub_lazy(a1, a2) = error("Not implemented.")
## function mul_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return lazy(Mul([a]))
##     elseif ismul(u)
##         return a
##     else
##         return error("Variant not supported.")
##     end
## end
## # Note that this is nested by default.
## mul_lazy(a1, a2) = lazy(Mul([a1, a2]))
## mul_lazy(a1::Number, a2) = error("Not implemented.")
## mul_lazy(a1, a2::Number) = error("Not implemented.")
## mul_lazy(a1::Number, a2::Number) = a1 * a2
## div_lazy(a1, a2::Number) = error("Not implemented.")
##
## # NamedDimsArrays.jl interface.
## function inds_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return inds(u)
##     elseif ismul(u)
##         return mapreduce(inds, symdiff, arguments(u))
##     else
##         return error("Variant not supported.")
##     end
## end
## function dename_lazy(a)
##     u = unwrap(a)
##     if !iscall(u)
##         return dename(u)
##     else
##         return error("Variant not supported.")
##     end
## end

## # Lazy broadcasting.
## struct LazyNamedDimsArrayStyle <: AbstractNamedDimsArrayStyle{Any} end
## function Broadcast.broadcasted(::LazyNamedDimsArrayStyle, f, as...)
##     return error("Arbitrary broadcasting not supported for LazyNamedDimsArray.")
## end
## # Linear operations.
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(+), a1, a2) = a1 + a2
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(-), a1, a2) = a1 - a2
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), c::Number, a) = c * a
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), a, c::Number) = a * c
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(*), a::Number, b::Number) = a * b
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(/), a, c::Number) = a / c
## Broadcast.broadcasted(::LazyNamedDimsArrayStyle, ::typeof(-), a) = -a

## # Generic functionality for Applied types, like `Mul`, `Add`, etc.
## ismul(a) = operation(a) ≡ *
## head_applied(a) = operation(a)
## iscall_applied(a) = true
## isexpr_applied(a) = iscall(a)
## function show_applied(io::IO, a)
##     args = map(arg -> sprint(AbstractTrees.printnode, arg), arguments(a))
##     print(io, "(", join(args, " $(operation(a)) "), ")")
##     return nothing
## end
## sorted_arguments_applied(a) = arguments(a)
## children_applied(a) = arguments(a)
## sorted_children_applied(a) = sorted_arguments(a)
## function maketerm_applied(type, head, args, metadata)
##     term = type(args)
##     @assert head ≡ operation(term)
##     return term
## end
## map_arguments_applied(f, a) = unspecify_type_parameters(typeof(a))(map(f, arguments(a)))
## function hash_applied(a, h::UInt64)
##     h = hash(Symbol(unspecify_type_parameters(typeof(a))), h)
##     for arg in arguments(a)
##         h = hash(arg, h)
##     end
##     return h
## end
##
## abstract type Applied end
## TermInterface.head(a::Applied) = head_applied(a)
## TermInterface.iscall(a::Applied) = iscall_applied(a)
## TermInterface.isexpr(a::Applied) = isexpr_applied(a)
## Base.show(io::IO, a::Applied) = show_applied(io, a)
## TermInterface.sorted_arguments(a::Applied) = sorted_arguments_applied(a)
## TermInterface.children(a::Applied) = children_applied(a)
## TermInterface.sorted_children(a::Applied) = sorted_children_applied(a)
## function TermInterface.maketerm(type::Type{<:Applied}, head, args, metadata)
##     return maketerm_applied(type, head, args, metadata)
## end
## map_arguments(f, a::Applied) = map_arguments_applied(f, a)
## Base.hash(a::Applied, h::UInt64) = hash_applied(a, h)
##
## struct Mul{A} <: Applied
##     arguments::Vector{A}
## end
## TermInterface.arguments(m::Mul) = getfield(m, :arguments)
## TermInterface.operation(m::Mul) = *

## @wrapped struct LazyNamedDimsArray{
##         T, A <: AbstractNamedDimsArray{T},
##     } <: AbstractNamedDimsArray{T, Any}
##     union::Union{A, Mul{LazyNamedDimsArray{T, A}}}
## end
## function LazyNamedDimsArray(a::AbstractNamedDimsArray)
##     # Use `eltype(typeof(a))` for arrays that have different
##     # runtime and compile time eltypes, like `ITensor`.
##     return LazyNamedDimsArray{eltype(typeof(a)), typeof(a)}(a)
## end
## function LazyNamedDimsArray(a::Mul{LazyNamedDimsArray{T, A}}) where {T, A}
##     return LazyNamedDimsArray{T, A}(a)
## end
## lazy(a::LazyNamedDimsArray) = a
## lazy(a::AbstractNamedDimsArray) = LazyNamedDimsArray(a)
## lazy(a::Mul{<:LazyNamedDimsArray}) = LazyNamedDimsArray(a)
##
## NamedDimsArrays.inds(a::LazyNamedDimsArray) = inds_lazy(a)
## NamedDimsArrays.dename(a::LazyNamedDimsArray) = dename_lazy(a)
##
## # Broadcasting
## function Base.BroadcastStyle(::Type{<:LazyNamedDimsArray})
##     return LazyNamedDimsArrayStyle()
## end
##
## # Derived functionality.
## function TermInterface.maketerm(type::Type{LazyNamedDimsArray}, head, args, metadata)
##     return maketerm_lazy(type, head, args, metadata)
## end
## Base.getindex(a::LazyNamedDimsArray, I::Int...) = getindex_lazy(a, I...)
## TermInterface.arguments(a::LazyNamedDimsArray) = arguments_lazy(a)
## TermInterface.children(a::LazyNamedDimsArray) = children_lazy(a)
## TermInterface.head(a::LazyNamedDimsArray) = head_lazy(a)
## TermInterface.iscall(a::LazyNamedDimsArray) = iscall_lazy(a)
## TermInterface.isexpr(a::LazyNamedDimsArray) = isexpr_lazy(a)
## TermInterface.operation(a::LazyNamedDimsArray) = operation_lazy(a)
## TermInterface.sorted_arguments(a::LazyNamedDimsArray) = sorted_arguments_lazy(a)
## AbstractTrees.children(a::LazyNamedDimsArray) = abstracttrees_children_lazy(a)
## TermInterface.sorted_children(a::LazyNamedDimsArray) = sorted_children_lazy(a)
## ismul(a::LazyNamedDimsArray) = ismul_lazy(a)
## AbstractTrees.nodevalue(a::LazyNamedDimsArray) = nodevalue_lazy(a)
## Base.Broadcast.materialize(a::LazyNamedDimsArray) = materialize_lazy(a)
## Base.copy(a::LazyNamedDimsArray) = copy_lazy(a)
## Base.:(==)(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = equals_lazy(a1, a2)
## Base.hash(a::LazyNamedDimsArray, h::UInt64) = hash_lazy(a, h)
## map_arguments(f, a::LazyNamedDimsArray) = map_arguments_lazy(f, a)
## substitute(a::LazyNamedDimsArray, substitutions) = substitute_lazy(a, substitutions)
## AbstractTrees.printnode(io::IO, a::LazyNamedDimsArray) = printnode_lazy(io, a)
## printnode_nameddims(io::IO, a::LazyNamedDimsArray) = printnode_lazy(io, a)
## Base.show(io::IO, a::LazyNamedDimsArray) = show_lazy(io, a)
## Base.show(io::IO, mime::MIME"text/plain", a::LazyNamedDimsArray) = show_lazy(io, mime, a)
## Base.:*(a::LazyNamedDimsArray) = mul_lazy(a)
## Base.:*(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = mul_lazy(a1, a2)
## Base.:+(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = add_lazy(a1, a2)
## Base.:-(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = sub_lazy(a1, a2)
## Base.:*(a1::Number, a2::LazyNamedDimsArray) = mul_lazy(a1, a2)
## Base.:*(a1::LazyNamedDimsArray, a2::Number) = mul_lazy(a1, a2)
## Base.:/(a1::LazyNamedDimsArray, a2::Number) = div_lazy(a1, a2)
## Base.:-(a::LazyNamedDimsArray) = sub_lazy(a)

## struct SymbolicArray{T, N, Name, Axes <: NTuple{N, AbstractUnitRange{<:Integer}}} <: AbstractArray{T, N}
##     name::Name
##     axes::Axes
##     function SymbolicArray{T}(name, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}}) where {T}
##         N = length(ax)
##         return new{T, N, typeof(name), typeof(ax)}(name, ax)
##     end
## end
## function SymbolicArray(name, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
##     return SymbolicArray{Any}(name, ax)
## end
## function SymbolicArray{T}(name, ax::AbstractUnitRange...) where {T}
##     return SymbolicArray{T}(name, ax)
## end
## function SymbolicArray(name, ax::AbstractUnitRange...)
##     return SymbolicArray{Any}(name, ax)
## end
## symname(a::SymbolicArray) = getfield(a, :name)
## Base.axes(a::SymbolicArray) = getfield(a, :axes)
## Base.size(a::SymbolicArray) = length.(axes(a))
## function Base.:(==)(a::SymbolicArray, b::SymbolicArray)
##     return symname(a) == symname(b) && axes(a) == axes(b)
## end
## function Base.hash(a::SymbolicArray, h::UInt64)
##     h = hash(:SymbolicArray, h)
##     h = hash(symname(a), h)
##     return hash(size(a), h)
## end
## function Base.getindex(a::SymbolicArray{<:Any, N}, I::Vararg{Int, N}) where {N}
##     return error("Indexing into SymbolicArray not supported.")
## end
## function Base.setindex!(a::SymbolicArray{<:Any, N}, value, I::Vararg{Int, N}) where {N}
##     return error("Indexing into SymbolicArray not supported.")
## end
## function Base.show(io::IO, mime::MIME"text/plain", a::SymbolicArray)
##     Base.summary(io, a)
##     println(io, ":")
##     print(io, repr(symname(a)))
##     return nothing
## end
## function Base.show(io::IO, a::SymbolicArray)
##     print(io, "SymbolicArray(", symname(a), ", ", size(a), ")")
##     return nothing
## end
## using AbstractTrees: AbstractTrees
## function AbstractTrees.printnode(io::IO, a::SymbolicArray)
##     print(io, repr(symname(a)))
##     return nothing
## end
## const SymbolicNamedDimsArray{T, N, Parent <: SymbolicArray{T, N}, DimNames} =
##     NamedDimsArray{T, N, Parent, DimNames}
## function symnameddims(name)
##     return lazy(NamedDimsArray(SymbolicArray(name), ()))
## end
## function AbstractTrees.printnode(io::IO, a::SymbolicNamedDimsArray)
##     print(io, symname(dename(a)))
##     if ndims(a) > 0
##         print(io, "[", join(dimnames(a), ","), "]")
##     end
##     return nothing
## end
## printnode_nameddims(io::IO, a::SymbolicNamedDimsArray) = AbstractTrees.printnode(io, a)
## function Base.:(==)(a::SymbolicNamedDimsArray, b::SymbolicNamedDimsArray)
##     return issetequal(inds(a), inds(b)) && dename(a) == dename(b)
## end
## Base.:*(a::SymbolicNamedDimsArray, b::SymbolicNamedDimsArray) = lazy(a) * lazy(b)
## Base.:*(a::SymbolicNamedDimsArray, b::LazyNamedDimsArray) = lazy(a) * b
## Base.:*(a::LazyNamedDimsArray, b::SymbolicNamedDimsArray) = a * lazy(b)

end
