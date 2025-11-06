module ITensorNetworksNextTensorOperationsExt

using BackendSelection: @Algorithm_str, Algorithm
using ITensorNetworksNext: ITensorNetworksNext, contraction_order
using NamedDimsArrays: inds
using TensorOperations: TensorOperations, optimaltree

function contraction_order_to_expr(seq)
    return seq isa AbstractVector ? prod(contraction_order_to_expr, seq) : symnameddims(seq)
end

function ITensorNetworksNext.contraction_order(alg::Algorithm"optimal", tn)
    ts = [tn[i] for i in eachindex(tn)]
    network = collect.(inds.(ts))
    # Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(length(i)) for i in unique(reduce(vcat, network)))
    seq, _ = optimaltree(network, inds_to_dims)
    # TODO: Map the integer indices back to the original tensor network vertices.
    expr = contraction_order_to_expr(seq)
    subs = Dict(symnameddims(i) => symnameddims(eachindex(tn)[i]) for i in eachindex(ts))
    return substitute(expr, subs)
end

end
