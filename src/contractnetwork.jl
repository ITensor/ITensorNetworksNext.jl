using BackendSelection: @Algorithm_str, Algorithm

default_sequence_alg = "leftassociative"

function contraction_sequence(::Algorithm"leftassociative", tn::Vector{<:AbstractArray})
  return Any[i for i in 1:length(tn)]
end

function contraction_sequence(tn::Vector{<:AbstractArray}; alg=default_sequence_alg)
  contraction_sequence(Algorithm(alg), tn)
end

# Internal recursive worker
function recursive_contractnetwork(tn::Union{AbstractVector,AbstractNamedDimsArray})
  tn isa AbstractVector && return prod(recursive_contractnetwork, tn)
  return tn
end

# Recursive worker for ordering the tensors according to the sequence
rearrange(tn::Vector{<:AbstractArray}, i::Integer) = tn[i]
rearrange(tn::Vector{<:AbstractArray}, v::AbstractVector) = [rearrange(tn, s) for s in v]

function contractnetwork(tn::Vector{<:AbstractArray}; sequence=default_sequence_alg)
  contract_sequence = isa(sequence, String) ? contraction_sequence(tn; alg=sequence) : sequence
  return recursive_contractnetwork(rearrange(tn, contract_sequence))
end

function contractnetwork(tn::AbstractTensorNetwork; sequence=default_sequence_alg)
  return contractnetwork([tn[v] for v in vertices(tn)]; sequence)
end
