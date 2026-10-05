using Base: @kwdef

"""
    Exact(; contraction_tree = nothing, tree_alg = Greedy())

Contract a network exactly, in the pairwise order given by `contraction_tree`. When no tree is
supplied one is found with `tree_alg`.
"""
@kwdef struct Exact{Tree, TreeAlg}
    contraction_tree::Tree = nothing
    tree_alg::TreeAlg = Greedy()
end

function contract_network(alg, tn)
    return throw(ArgumentError("`contract_network` algorithm `$(alg)` not implemented."))
end
contract_network(tn; alg = Exact()) = contract_network(alg, tn)

function contract_network(alg::Exact, tn)
    tree = @something alg.contraction_tree contraction_tree(tn; alg = alg.tree_alg)
    return prod_tensors(tn, tree)
end

# A `NormNetwork` has no tensor at a vertex, only the two layers that would form it, so it is
# contracted through `flatten_network`. Supplying a tree alongside it is not meaningful, since the
# tree would have to be keyed by the flattened vertices: pass `flatten_network(nn)` instead, and
# the keys are its own vertices as for any other network.
function contract_network(alg::Exact, nn::NormNetwork)
    isnothing(alg.contraction_tree) || throw(
        ArgumentError(
            "A contraction tree cannot be keyed by the vertices of a `NormNetwork`, whose vertices carry no tensor. Contract `flatten_network(nn)` instead, whose vertices the tree can be built over."
        )
    )
    return contract_network(alg, flatten_network(nn))
end
