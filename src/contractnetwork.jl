using BackendSelection: @Algorithm_str, Algorithm
using ITensorNetworksNext.LazyNamedDimsArrays: nested_array_to_lazy_multiply, substitute_lazy, materialize, lazy,
    symnameddims

default_contract_alg = nothing

#Algorithmic defaults
default_sequence(::Algorithm"exact") = "leftassociative"
function set_default_kwargs(alg::Algorithm"exact")
    sequence = get(alg, :sequence, default_sequence(alg))
    return Algorithm("exact"; sequence)
end

function contraction_sequence(::Algorithm"leftassociative", tn::Vector{<:AbstractArray})
    return nested_array_to_lazy_multiply(collect.(1:length(tn)))
end

function contraction_sequence(tn::Vector{<:AbstractArray}; alg = default_sequence_alg)
    return contraction_sequence(Algorithm(alg), tn)
end

function contractnetwork(alg::Algorithm"exact", tn::Vector{<:AbstractArray})
    contract_sequence = isa(alg.sequence, String) ? contraction_sequence(tn; alg = alg.sequence) : sequence
    contract_sequence = substitute_lazy(contract_sequence, Dict(symnameddims(i) => lazy(tn[i]) for i in 1:length(tn)))
    return materialize(contract_sequence)
end

function contractnetwork(alg::Algorithm"exact", tn::AbstractTensorNetwork)
    return contractnetwork(alg, [tn[v] for v in vertices(tn)])
end

function contractnetwork(tn::Union{AbstractTensorNetwork, Vector{<:AbstractArray}}; alg = default_contract_alg, kwargs...)
    alg == nothing && error("Must specify an algorithm to contract the network with")
    return contractnetwork(set_default_kwargs(Algorithm(alg; kwargs...)), tn)
end
