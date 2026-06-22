module ITensorNetworksNextTensorOperationsExt

using BackendSelection: @Algorithm_str, Algorithm
using ITensorBase: denamed, inds
using ITensorNetworksNext.LazyITensors.TermInterface: arguments
using ITensorNetworksNext.LazyITensors: LazyITensors, ismul, substitute, symnameddims
using TensorOperations: TensorOperations, optimaltree

function contraction_tree_to_expr(f, tree)
    return if !(tree isa AbstractVector)
        f(tree)
    else
        prod(Base.Fix1(contraction_tree_to_expr, f), tree)
    end
end

function LazyITensors.optimize_contraction_order(alg::Algorithm"optimal", a)
    @assert ismul(a)
    ts = arguments(a)
    inds_network = collect.(inds.(ts))
    # Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(length(denamed(i))) for i in reduce(∪, inds_network))
    tree, _ = optimaltree(inds_network, inds_to_dims)
    return contraction_tree_to_expr(i -> ts[i], tree)
end

end
