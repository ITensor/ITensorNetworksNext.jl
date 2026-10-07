module ITensorNetworksNextOMEinsumContractionOrdersExt

using ITensorBase: inds, name
using ITensorNetworksNext:
    ITensorNetworksNext, ContractionTree, contraction_tree, prod_tensors
using OMEinsumContractionOrders:
    OMEinsumContractionOrders, CodeOptimizer, EinCode, NestedEinsum, optimize_code

# Rebuild a `ContractionTree` from an optimized `NestedEinsum`, mapping each leaf's
# `tensorindex` back to the key it came from.
function nested_einsum_to_tree(ks, code::NestedEinsum)
    # A leaf holds the 1-based index of its input tensor; internal nodes hold `-1`.
    code.tensorindex != -1 && return ContractionTree(ks[code.tensorindex])
    return reduce(ContractionTree, map(Base.Fix1(nested_einsum_to_tree, ks), code.args))
end

# Find a contraction order with any OMEinsumContractionOrders optimizer (`GreedyMethod`,
# `TreeSA`, `KaHyParBipartite`, ...) by forwarding to `optimize_code`.
function ITensorNetworksNext.contraction_tree(alg::CodeOptimizer, tensors)
    ks = collect(keys(tensors))
    ixs = [map(name, inds(tensors[k])) for k in ks]
    all_inds = reduce(vcat, ixs)
    labels = unique(all_inds)
    size_dict = Dict(name(i) => length(i) for k in ks for i in inds(tensors[k]))
    # Open indices (appearing on a single tensor) are the output of the network.
    iy = filter(i -> count(==(i), all_inds) == 1, labels)
    code = optimize_code(EinCode(ixs, iy), size_dict, alg)
    return nested_einsum_to_tree(ks, code)
end

# `CodeOptimizer` cannot subtype `ContractionTreeAlgorithm`, so the order-taking form of
# `prod_tensors` is extended for it here.
function ITensorNetworksNext.prod_tensors(tensors, alg::CodeOptimizer)
    return prod_tensors(tensors, contraction_tree(alg, tensors))
end

end
