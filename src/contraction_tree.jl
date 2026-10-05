using AbstractTrees: AbstractTrees
using WrappedUnions: @wrapped, unwrap

"""
    ContractionTree{V}

A fully-resolved binary contraction tree over vertex labels of type `V`. A node is either a
leaf holding a single vertex label, or a branch holding a 2-tuple of child trees; a branch
records that its two children are contracted together, and nesting fixes the pairwise order.

`ContractionTree` is inert structure, not a lazy contraction: it carries no operation and does
not know how to contract itself. Actually contracting a network maps the leaf labels to tensors
and folds the tree elsewhere; the tree only stores the labels and their nesting.

Construct a leaf from a label and a branch from two subtrees:

```julia
ContractionTree(:a)                                        # leaf
ContractionTree(ContractionTree(:a), ContractionTree(:b))  # branch
```

Navigate with `isleaf`, integer indexing (`t[1]`, `t[2]`), and the `AbstractTrees` interface
(`children`, `nodevalue`).
"""
@wrapped struct ContractionTree{V}
    # A branch's children; equal to `Branch{V}` below, spelled out here because the alias
    # references `ContractionTree` and so cannot be defined before this struct.
    union::Union{V, NTuple{2, ContractionTree{V}}}
end

# The children of a branch: a 2-tuple of subtrees. Aliased so the branch member type is named
# everywhere except its definitional site (the struct field above).
const Branch{V} = NTuple{2, ContractionTree{V}}

# A leaf is built by the default constructor `ContractionTree(label)`. A branch is built from a
# 2-tuple of subtrees; that form is more specific than the default, so `ContractionTree((l, r))`
# builds a branch, not a leaf whose label is a tuple, and `ContractionTree(l, r)` is shorthand.
function ContractionTree(children::Branch{V}) where {V}
    return ContractionTree{V}(children)
end
function ContractionTree(left::ContractionTree{V}, right::ContractionTree{V}) where {V}
    return ContractionTree{V}((left, right))
end

# A branch wraps a 2-tuple of child trees; a leaf wraps its vertex label. Dispatching on the
# branch member type parametrized by `V` (not a bare `Tuple`) keeps this correct even when the
# vertex labels are themselves tuples.
isleaf(t::ContractionTree{V}) where {V} = !(unwrap(t) isa Branch{V})

# `nodevalue` is a leaf's vertex label; only a branch has children.
AbstractTrees.nodevalue(t::ContractionTree) = unwrap(t)
AbstractTrees.children(t::ContractionTree) = isleaf(t) ? () : unwrap(t)

Base.getindex(t::ContractionTree, i::Int) = unwrap(t)[i]

Base.:(==)(a::ContractionTree, b::ContractionTree) = unwrap(a) == unwrap(b)
Base.hash(t::ContractionTree, h::UInt) = hash(unwrap(t), hash(:ContractionTree, h))

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
