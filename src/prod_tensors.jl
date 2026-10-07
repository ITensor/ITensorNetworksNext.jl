using AbstractTrees: nodevalue
using Combinatorics: combinations
using ITensorBase: inds

"""
    ContractionTreeAlgorithm

Supertype of the strategies for finding a contraction order. See [`contraction_tree`](@ref).
"""
abstract type ContractionTreeAlgorithm <: AbstractAlgorithm end

"""
    prod_tensors(tensors, tree::ContractionTree)
    prod_tensors(tensors, alg)
    prod_tensors(tensors)

Contract `tensors` in the pairwise order given by `tree`, whose leaf labels index into
`tensors`. Given an order algorithm instead, find a tree with it first. Given neither, contract
in key order, folding from the left.

`tensors` is anything indexable by those labels, so a `Vector` pairs with a tree over positions
and a tensor network pairs with a tree over vertices.
"""
function prod_tensors end
prod_tensors(tensors) = prod_tensors(tensors, left_associative_tree(keys(tensors)))
function prod_tensors(tensors, tree::ContractionTree)
    isleaf(tree) && return tensors[nodevalue(tree)]
    return prod_tensors(tensors, tree[1]) * prod_tensors(tensors, tree[2])
end
function prod_tensors(tensors, alg::ContractionTreeAlgorithm)
    return prod_tensors(tensors, contraction_tree(alg, tensors))
end

# The tree that folds `labels` from the left, so `(a, b, c)` gives `((a, b), c)`.
function left_associative_tree(labels)
    isempty(labels) && throw(ArgumentError("No tensors to contract."))
    return reduce(ContractionTree, map(ContractionTree, labels))
end

"""
    contraction_tree(tensors; alg = Greedy())
    contraction_tree(alg, tensors)

Find a pairwise contraction order for `tensors`, returned as a [`ContractionTree`](@ref) over
their keys.

Only the index names and lengths are consulted, never the entries, so an order found once can
be replayed against any network of the same shape.
"""
function contraction_tree end
contraction_tree(tensors; alg = Greedy()) = contraction_tree(alg, tensors)
function contraction_tree(alg, tensors)
    return throw(ArgumentError("Contraction order algorithm `$(alg)` not implemented."))
end

"""
    Greedy

Repeatedly contract the cheapest available pair, measured as the product of the lengths of all
indices involved. Outer products are taken only once nothing else is left.
"""
struct Greedy <: ContractionTreeAlgorithm end

function contraction_tree(::Greedy, tensors)
    ks = collect(keys(tensors))
    isempty(ks) && throw(ArgumentError("No tensors to contract."))
    trees = map(ContractionTree, ks)
    # The inds of each pending operand. An intermediate never materializes, so its inds are
    # tracked here rather than read off a tensor.
    is = [inds(tensors[k]) for k in ks]
    while length(trees) > 1
        i1, i2 = argmin(combinations(eachindex(trees), 2)) do (i, j)
            # Defer outer products: with nothing contracted they cost the full product of both
            # operands, and taking one early inflates every contraction that follows.
            isdisjoint(is[i], is[j]) && return typemax(Int)
            # Every index on either operand is iterated once, shared ones counted once.
            return prod(length, union(is[i], is[j]); init = 1)
        end
        # Shared indices are summed over, so the result carries the symmetric difference.
        contracted = symdiff(is[i1], is[i2])
        tree = ContractionTree(trees[i1], trees[i2])
        # Remove the pair by position. Removing it by value would also drop any other operand
        # equal to it, silently losing a tensor from a network with repeated tensors.
        keep = [i for i in eachindex(trees) if i ∉ (i1, i2)]
        trees = [trees[keep]; [tree]]
        is = [is[keep]; [contracted]]
    end
    return only(trees)
end
