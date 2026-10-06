using AbstractTrees: AbstractTrees

"""
    Branch(left, right)

The children of a branch node, as held by a [`ContractionTree`](@ref).

Giving the children a type of their own is what keeps a branch distinguishable from a leaf: a
vertex label can be a 2-tuple, so a bare tuple of children would be ambiguous, but no label is
a `Branch` unless it is deliberately made one.
"""
struct Branch{T}
    children::NTuple{2, T}
end
Branch(left, right) = Branch((left, right))

"""
    ContractionTree{V}

A binary tree representing a contraction order over leaf labels of type `V`. A node is either a
leaf holding one label or a [`Branch`](@ref) holding its two child trees.
"""
struct ContractionTree{V}
    node::Union{V, Branch{ContractionTree{V}}}
end

function ContractionTree(children::Branch{ContractionTree{V}}) where {V}
    return ContractionTree{V}(children)
end
function ContractionTree(left::ContractionTree{V}, right::ContractionTree{V}) where {V}
    return ContractionTree{V}(Branch(left, right))
end

isleaf(t::ContractionTree) = !(t.node isa Branch)

AbstractTrees.nodevalue(t::ContractionTree) = t.node
AbstractTrees.children(t::ContractionTree) = isleaf(t) ? () : t.node.children

Base.getindex(t::ContractionTree, i::Int) = t.node.children[i]

Base.:(==)(a::ContractionTree, b::ContractionTree) = a.node == b.node
Base.hash(t::ContractionTree, h::UInt) = hash(t.node, hash(:ContractionTree, h))

function Base.show(io::IO, t::ContractionTree)
    if isleaf(t)
        show(io, AbstractTrees.nodevalue(t))
    else
        print(io, "(")
        show(io, t[1])
        print(io, ", ")
        show(io, t[2])
        print(io, ")")
    end
    return nothing
end
function AbstractTrees.printnode(io::IO, t::ContractionTree)
    isleaf(t) && show(io, AbstractTrees.nodevalue(t))
    return nothing
end
