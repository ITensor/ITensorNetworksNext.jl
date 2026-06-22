using ITensorBase: ITensorBase, AbstractITensor
using WrappedUnions: @wrapped

@wrapped struct LazyITensor{
        DimName, A <: AbstractITensor{DimName},
    } <: AbstractITensor{DimName}
    union::Union{A, Mul{LazyITensor{DimName, A}}}
end

parenttype(::Type{LazyITensor{DimName, A}}) where {DimName, A} = A
parenttype(::Type{LazyITensor{DimName}}) where {DimName} = AbstractITensor{DimName}
parenttype(::Type{LazyITensor}) = AbstractITensor

function LazyITensor(a::AbstractITensor)
    return LazyITensor{ITensorBase.dimnametype(typeof(a)), typeof(a)}(a)
end
function LazyITensor(a::Mul{L}) where {L <: LazyITensor}
    return LazyITensor{ITensorBase.dimnametype(L), parenttype(L)}(a)
end
lazy(a::LazyITensor) = a
lazy(a::AbstractITensor) = LazyITensor(a)
lazy(a::Mul{<:LazyITensor}) = LazyITensor(a)

ITensorBase.dimnames(a::LazyITensor) = dimnames_lazy(a)
ITensorBase.inds(a::LazyITensor) = inds_lazy(a)
ITensorBase.denamed(a::LazyITensor) = denamed_lazy(a)

# Broadcasting
function Base.BroadcastStyle(::Type{<:LazyITensor})
    return LazyITensorStyle()
end

# Derived functionality.
function TermInterface.maketerm(type::Type{LazyITensor}, head, args, metadata)
    return maketerm_lazy(type, head, args, metadata)
end
Base.getindex(a::LazyITensor, I::Int...) = getindex_lazy(a, I...)
TermInterface.arguments(a::LazyITensor) = arguments_lazy(a)
TermInterface.children(a::LazyITensor) = children_lazy(a)
TermInterface.head(a::LazyITensor) = head_lazy(a)
TermInterface.iscall(a::LazyITensor) = iscall_lazy(a)
TermInterface.isexpr(a::LazyITensor) = isexpr_lazy(a)
TermInterface.operation(a::LazyITensor) = operation_lazy(a)
TermInterface.sorted_arguments(a::LazyITensor) = sorted_arguments_lazy(a)
AbstractTrees.children(a::LazyITensor) = abstracttrees_children_lazy(a)
TermInterface.sorted_children(a::LazyITensor) = sorted_children_lazy(a)
ismul(a::LazyITensor) = ismul_lazy(a)
AbstractTrees.nodevalue(a::LazyITensor) = nodevalue_lazy(a)
Base.Broadcast.materialize(a::LazyITensor) = materialize_lazy(a)
Base.copy(a::LazyITensor) = copy_lazy(a)
Base.:(==)(a1::LazyITensor, a2::LazyITensor) = equals_lazy(a1, a2)
Base.isequal(a1::LazyITensor, a2::LazyITensor) = isequal_lazy(a1, a2)
Base.hash(a::LazyITensor, h::UInt64) = hash_lazy(a, h)
map_arguments(f, a::LazyITensor) = map_arguments_lazy(f, a)
substitute(a::LazyITensor, substitutions) = substitute_lazy(a, substitutions)
AbstractTrees.printnode(io::IO, a::LazyITensor) = printnode_lazy(io, a)
printnode_nameddims(io::IO, a::LazyITensor) = printnode_lazy(io, a)
Base.show(io::IO, a::LazyITensor) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::LazyITensor) = show_lazy(io, mime, a)
Base.:*(a::LazyITensor) = mul_lazy(a)
Base.:*(a1::LazyITensor, a2::LazyITensor) = mul_lazy(a1, a2)
Base.:+(a1::LazyITensor, a2::LazyITensor) = add_lazy(a1, a2)
Base.:-(a1::LazyITensor, a2::LazyITensor) = sub_lazy(a1, a2)
Base.:*(a1::Number, a2::LazyITensor) = mul_lazy(a1, a2)
Base.:*(a1::LazyITensor, a2::Number) = mul_lazy(a1, a2)
Base.:/(a1::LazyITensor, a2::Number) = div_lazy(a1, a2)
Base.:-(a::LazyITensor) = sub_lazy(a)
