module SymbolicArrays

using AbstractTrees: AbstractTrees

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
name(a::SymbolicArray) = getfield(a, :name)
Base.axes(a::SymbolicArray) = getfield(a, :axes)
Base.size(a::SymbolicArray) = length.(axes(a))
function Base.:(==)(a::SymbolicArray, b::SymbolicArray)
    return name(a) == name(b) && axes(a) == axes(b)
end
function Base.show(io::IO, mime::MIME"text/plain", a::SymbolicArray)
    Base.summary(io, a)
    println(io, ":")
    print(io, repr(name(a)))
    return nothing
end
function Base.show(io::IO, a::SymbolicArray)
    print(io, "SymbolicArray(", name(a), ", ", size(a), ")")
    return nothing
end

function AbstractTrees.printnode(io::IO, a::SymbolicArray)
    print(io, repr(name(a)))
    return nothing
end

end
