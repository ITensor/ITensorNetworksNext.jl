using BackendSelection: @Algorithm_str, Algorithm

default_contract_alg = nothing

#Algorithmic defaults
default_sequence(::Algorithm"exact") = "leftassociative"
function set_default_kwargs(alg::Algorithm"exact")
  sequence = get(alg, :sequence, default_sequence(alg))
  return Algorithm("exact"; sequence)
end

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

function contractnetwork(alg::Algorithm"exact", tn::Vector{<:AbstractArray})
  contract_sequence = isa(alg.sequence, String) ? contraction_sequence(tn; alg=alg.sequence) : sequence
  return recursive_contractnetwork(rearrange(tn, contract_sequence))
end

function contractnetwork(alg::Algorithm"exact", tn::AbstractTensorNetwork)
  return contractnetwork(alg, [tn[v] for v in vertices(tn)])
end

function contractnetwork(tn::Union{AbstractTensorNetwork, Vector{<:AbstractArray}}; alg = default_contract_alg, kwargs...)
  alg == nothing && error("Must specify an algorithm to contract the network with")
  return contractnetwork(set_default_kwargs(Algorithm(alg; kwargs...)), tn)
end
