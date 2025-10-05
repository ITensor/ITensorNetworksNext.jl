module ITensorNetworksTensorOperationsExt

using ITensorNetworksNext: ITensorNetworksNext
using TensorOperations: TensorOperations, optimaltree

function ITensorNetworksNext.contraction_sequence(::Algorithm"optimal", tn::Vector{<:AbstractArray})
    network = collect.(inds.(tn))
    #Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(reduce(vcat, network)))
    seq, _ = optimaltree(network, inds_to_dims)
    return seq
end

end