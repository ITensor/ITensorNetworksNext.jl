module ITensorNetworksNextTensorOperationsExt

using BackendSelection: @Algorithm_str, Algorithm
using NamedDimsArrays: inds
using ITensorNetworksNext: ITensorNetworksNext, contraction_sequence_to_expr
using TensorOperations: TensorOperations, optimaltree

function ITensorNetworksNext.contraction_sequence(::Algorithm"optimal", tn::Vector{<:AbstractArray})
    network = collect.(inds.(tn))
    #Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(length(i)) for i in unique(reduce(vcat, network)))
    seq, _ = optimaltree(network, inds_to_dims)
    return contraction_sequence_to_expr(seq)
end

end
