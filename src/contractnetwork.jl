using TensorOperations: TensorOperations, optimaltree
using ITensorBase: inds, dim

default_sequence_alg = "optimal"

function contraction_sequence(::Algorithm"optimal", tn::Vector{<:AbstractArray})
  network = collect.(inds.(tn))
  #Converting dims to Float64 to minimize overflow issues
  inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(reduce(vcat, network)))
  seq, _ = optimaltree(network, inds_to_dims)
  return seq
end

function contraction_sequence(::Algorithm"leftassociative", tn::Vector{<:AbstractArray})
  return Any[i for i in 1:length(tn)]
end

function contraction_sequence(tn::Vector{<:AbstractArray}; alg=default_sequence_alg)
  contraction_sequence(Algorithm(alg), tn)
end

# Internal recursive worker
function recursive_contractnetwork(tn::Union{AbstractVector,AbstractArray})
  tn isa AbstractVector && return reduce(*, map(recursive_contractnetwork, tn))
  return tn
end

# Recursive worker for ordering the tensors according to the sequence
rearrange(tn::Vector{<:AbstractArray}, i::Integer) = tn[i]
rearrange(tn::Vector{<:AbstractArray}, v::AbstractVector) = [rearrange(tn, s) for s in v]

function contractnetwork(tn::Vector{<:AbstractArray}; sequence_alg=default_sequence_alg)
  sequence = contraction_sequence(tn; alg=sequence_alg)
  return recursive_contractnetwork(rearrange(tn, sequence))
end

function contractnetwork(tn::AbstractTensorNetwork; sequence_alg=default_sequence_alg)
  return contractnetwork([tn[v] for v in vertices(tn)]; sequence_alg)
end
