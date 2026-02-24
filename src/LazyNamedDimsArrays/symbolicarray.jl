# TODO: Allow dynamic/unknown number of dimensions by supporting vector axes.
struct SymbolicArray{T, N, Name, Axes <: NTuple{N, AbstractUnitRange{<:Integer}}} <:
    AbstractArray{T, N}
    name::Name
    axes::Axes
    function SymbolicArray{T}(
            name,
            ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}}
        ) where {T}
        N = length(ax)
        return new{T, N, typeof(name), typeof(ax)}(name, ax)
    end
end
function SymbolicArray(name, ax::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    return SymbolicArray{Any}(name, ax)
end
symname(a::SymbolicArray) = getfield(a, :name)
Base.axes(a::SymbolicArray) = getfield(a, :axes)
Base.size(a::SymbolicArray) = length.(axes(a))
function Base.:(==)(a::SymbolicArray, b::SymbolicArray)
    return symname(a) == symname(b) && axes(a) == axes(b)
end
Base.isequal(a::SymbolicArray, b::SymbolicArray) = a == b
function Base.hash(a::SymbolicArray, h::UInt64)
    h = hash(:SymbolicArray, h)
    h = hash(symname(a), h)
    return hash(size(a), h)
end
function Base.getindex(a::SymbolicArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    return error("Indexing into SymbolicArray not supported.")
end
function Base.setindex!(a::SymbolicArray{<:Any, N}, value, I::Vararg{Int, N}) where {N}
    return error("Indexing into SymbolicArray not supported.")
end
using FunctionImplementations: FunctionImplementations
FunctionImplementations.permuteddims(a::SymbolicArray, p) = permutedims(a, p)
function Base.permutedims(a::SymbolicArray, p)
    @assert ndims(a) == length(p) && isperm(p)
    return SymbolicArray(symname(a), ntuple(i -> axes(a)[p[i]], ndims(a)))
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
