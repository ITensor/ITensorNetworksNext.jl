module ITensorNetworksNextTensorOperationsExt

using BackendSelection: @Algorithm_str, Algorithm
using ITensorNetworksNext: ITensorNetworksNext, contraction_order
using ITensorNetworksNext.LazyNamedDimsArrays: symnameddims, substitute
using NamedDimsArrays: inds
using TensorOperations: TensorOperations, optimaltree

function contraction_order_to_expr(ord)
    return ord isa AbstractVector ? prod(contraction_order_to_expr, ord) : symnameddims(ord)
end

function ITensorNetworksNext.contraction_order(alg::Algorithm"optimal", tn)
    ts = [tn[i] for i in keys(tn)]
    network = collect.(inds.(ts))
    # Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(length(i)) for i in unique(reduce(vcat, network)))
    order, _ = optimaltree(network, inds_to_dims)
    # TODO: Map the integer indices back to the original tensor network vertices.
    expr = contraction_order_to_expr(order)
    verts = collect(keys(tn))
    sym(i) = symnameddims(verts[i], Tuple(inds(tn[verts[i]])))
    subs = Dict(symnameddims(i) => sym(i) for i in eachindex(verts))
    return substitute(expr, subs)
end

end
