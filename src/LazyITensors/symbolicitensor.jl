using ITensorBase: ITensorBase, AbstractITensor, dimnames, inds, name

# Expression leaf with no array payload, so it defines no `denamed`/`getindex`.
# Parameterized on `DimName` only (axes stored as a field, not a type parameter)
# so mixed-rank symbolic tensors share one concrete type and a flat `Mul` over
# them stays concretely typed.
struct SymbolicITensor{DimName} <: AbstractITensor{DimName}
    name::Any
    inds::Any
end
function SymbolicITensor(name, inds)
    DimName = isempty(inds) ? typeof(name) : eltype(ITensorBase.name.(inds))
    return SymbolicITensor{DimName}(name, inds)
end

symname(a::SymbolicITensor) = getfield(a, :name)

ITensorBase.dimnames(a::SymbolicITensor) = collect(ITensorBase.name.(getfield(a, :inds)))
ITensorBase.inds(a::SymbolicITensor) = getfield(a, :inds)
ITensorBase.dimnametype(::Type{<:SymbolicITensor{DimName}}) where {DimName} = DimName
Base.ndims(a::SymbolicITensor) = length(getfield(a, :inds))

function Base.:(==)(a::SymbolicITensor, b::SymbolicITensor)
    return symname(a) == symname(b) && dimnames(a) == dimnames(b)
end
Base.isequal(a::SymbolicITensor, b::SymbolicITensor) = a == b
function Base.hash(a::SymbolicITensor, h::UInt64)
    h = hash(:SymbolicITensor, h)
    h = hash(symname(a), h)
    return hash(dimnames(a), h)
end

# Products build lazy expressions rather than contracting numerically.
Base.:*(a::SymbolicITensor, b::SymbolicITensor) = lazy(a) * lazy(b)
Base.:*(a::SymbolicITensor, b::LazyITensor) = lazy(a) * b
Base.:*(a::LazyITensor, b::SymbolicITensor) = a * lazy(b)

issymbolic(a) = a isa SymbolicITensor
issymbolic(a::LazyITensor) = !iscall(a) && issymbolic(unwrap(a))

function Base.show(io::IO, a::SymbolicITensor)
    print(io, symname(a))
    if ndims(a) > 0
        print(io, "[", join(dimnames(a), ","), "]")
    end
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::SymbolicITensor)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

using AbstractTrees: AbstractTrees
function AbstractTrees.printnode(io::IO, a::SymbolicITensor)
    show(io, a)
    return nothing
end

function symnameddims(symname, dims)
    return lazy(SymbolicITensor(symname, dims))
end
symnameddims(name) = symnameddims(name, ())

function printnode_nameddims(io::IO, a::SymbolicITensor)
    AbstractTrees.printnode(io, a)
    return nothing
end
