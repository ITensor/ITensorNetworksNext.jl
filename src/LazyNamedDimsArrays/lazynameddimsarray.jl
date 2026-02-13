using NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray
using WrappedUnions: @wrapped

@wrapped struct LazyNamedDimsArray{
        T, A <: AbstractNamedDimsArray{T},
    } <: AbstractNamedDimsArray{T, Any}
    union::Union{A, Mul{LazyNamedDimsArray{T, A}}}
end

parenttype(::Type{LazyNamedDimsArray{T, A}}) where {T, A} = A
parenttype(::Type{LazyNamedDimsArray{T}}) where {T} = AbstractNamedDimsArray{T}
parenttype(::Type{LazyNamedDimsArray}) = AbstractNamedDimsArray

function LazyNamedDimsArray(a::AbstractNamedDimsArray)
    # Use `eltype(typeof(a))` for arrays that have different
    # runtime and compile time eltypes, like `ITensor`.
    return LazyNamedDimsArray{eltype(typeof(a)), typeof(a)}(a)
end
function LazyNamedDimsArray(a::Mul{L}) where {L <: LazyNamedDimsArray}
    return LazyNamedDimsArray{eltype(L), parenttype(L)}(a)
end
lazy(a::LazyNamedDimsArray) = a
lazy(a::AbstractNamedDimsArray) = LazyNamedDimsArray(a)
lazy(a::Mul{<:LazyNamedDimsArray}) = LazyNamedDimsArray(a)

NamedDimsArrays.dimnames(a::LazyNamedDimsArray) = dimnames_lazy(a)
NamedDimsArrays.inds(a::LazyNamedDimsArray) = inds_lazy(a)
NamedDimsArrays.denamed(a::LazyNamedDimsArray) = denamed_lazy(a)

# Broadcasting
function Base.BroadcastStyle(::Type{<:LazyNamedDimsArray})
    return LazyNamedDimsArrayStyle()
end

# Derived functionality.
function TermInterface.maketerm(type::Type{LazyNamedDimsArray}, head, args, metadata)
    return maketerm_lazy(type, head, args, metadata)
end
Base.getindex(a::LazyNamedDimsArray, I::Int...) = getindex_lazy(a, I...)
TermInterface.arguments(a::LazyNamedDimsArray) = arguments_lazy(a)
TermInterface.children(a::LazyNamedDimsArray) = children_lazy(a)
TermInterface.head(a::LazyNamedDimsArray) = head_lazy(a)
TermInterface.iscall(a::LazyNamedDimsArray) = iscall_lazy(a)
TermInterface.isexpr(a::LazyNamedDimsArray) = isexpr_lazy(a)
TermInterface.operation(a::LazyNamedDimsArray) = operation_lazy(a)
TermInterface.sorted_arguments(a::LazyNamedDimsArray) = sorted_arguments_lazy(a)
AbstractTrees.children(a::LazyNamedDimsArray) = abstracttrees_children_lazy(a)
TermInterface.sorted_children(a::LazyNamedDimsArray) = sorted_children_lazy(a)
ismul(a::LazyNamedDimsArray) = ismul_lazy(a)
AbstractTrees.nodevalue(a::LazyNamedDimsArray) = nodevalue_lazy(a)
Base.Broadcast.materialize(a::LazyNamedDimsArray) = materialize_lazy(a)
Base.copy(a::LazyNamedDimsArray) = copy_lazy(a)
Base.:(==)(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = equals_lazy(a1, a2)
Base.isequal(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = isequal_lazy(a1, a2)
Base.hash(a::LazyNamedDimsArray, h::UInt64) = hash_lazy(a, h)
map_arguments(f, a::LazyNamedDimsArray) = map_arguments_lazy(f, a)
substitute(a::LazyNamedDimsArray, substitutions) = substitute_lazy(a, substitutions)
AbstractTrees.printnode(io::IO, a::LazyNamedDimsArray) = printnode_lazy(io, a)
printnode_nameddims(io::IO, a::LazyNamedDimsArray) = printnode_lazy(io, a)
Base.show(io::IO, a::LazyNamedDimsArray) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::LazyNamedDimsArray) = show_lazy(io, mime, a)
Base.:*(a::LazyNamedDimsArray) = mul_lazy(a)
Base.:*(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = mul_lazy(a1, a2)
Base.:+(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = add_lazy(a1, a2)
Base.:-(a1::LazyNamedDimsArray, a2::LazyNamedDimsArray) = sub_lazy(a1, a2)
Base.:*(a1::Number, a2::LazyNamedDimsArray) = mul_lazy(a1, a2)
Base.:*(a1::LazyNamedDimsArray, a2::Number) = mul_lazy(a1, a2)
Base.:/(a1::LazyNamedDimsArray, a2::Number) = div_lazy(a1, a2)
Base.:-(a::LazyNamedDimsArray) = sub_lazy(a)
