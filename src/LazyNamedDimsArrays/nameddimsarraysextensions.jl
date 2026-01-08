using NamedDimsArrays: NamedDimsArray, denamed, inds
# Defined to avoid type piracy.
# TODO: Define a proper hash function
# in NamedDimsArrays.jl, maybe one that is
# independent of the order of dimensions.
function _hash(a::NamedDimsArray, h::UInt64)
    h = hash(:NamedDimsArray, h)
    h = hash(denamed(a), h)
    for i in inds(a)
        h = hash(i, h)
    end
    return h
end
function _hash(x, h::UInt64)
    return hash(x, h)
end

using AbstractTrees: AbstractTrees
using NamedDimsArrays: AbstractNamedDimsArray, dimnames
# Custom version of `AbstractTrees.printnode` to
# avoid type piracy when overloading on `AbstractNamedDimsArray`.
printnode_nameddims(io::IO, x) = AbstractTrees.printnode(io, x)
function printnode_nameddims(io::IO, a::AbstractNamedDimsArray)
    show(io, collect(dimnames(a)))
    return nothing
end
