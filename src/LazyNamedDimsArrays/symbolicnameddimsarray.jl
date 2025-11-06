using NamedDimsArrays: NamedDimsArray, dename, inds

const SymbolicNamedDimsArray{T, N, Parent <: SymbolicArray{T, N}, DimNames} =
    NamedDimsArray{T, N, Parent, DimNames}
function symnameddims(name)
    return lazy(NamedDimsArray(SymbolicArray(name), ()))
end
using AbstractTrees: AbstractTrees
function AbstractTrees.printnode(io::IO, a::SymbolicNamedDimsArray)
    print(io, symname(dename(a)))
    if ndims(a) > 0
        print(io, "[", join(dimnames(a), ","), "]")
    end
    return nothing
end
printnode_nameddims(io::IO, a::SymbolicNamedDimsArray) = AbstractTrees.printnode(io, a)
function Base.:(==)(a::SymbolicNamedDimsArray, b::SymbolicNamedDimsArray)
    return issetequal(inds(a), inds(b)) && dename(a) == dename(b)
end
Base.:*(a::SymbolicNamedDimsArray, b::SymbolicNamedDimsArray) = lazy(a) * lazy(b)
Base.:*(a::SymbolicNamedDimsArray, b::LazyNamedDimsArray) = lazy(a) * b
Base.:*(a::LazyNamedDimsArray, b::SymbolicNamedDimsArray) = a * lazy(b)
