using BackendSelection: @Algorithm_str, Algorithm
using ITensorNetworksNext.LazyNamedDimsArrays: substitute, materialize, lazy,
    symnameddims, substitute_lazy

#Algorithmic defaults
default_sequence(::Algorithm"exact") = "leftassociative"
function set_default_kwargs(alg::Algorithm"exact")
    sequence = get(alg, :sequence, default_sequence(alg))
    return Algorithm("exact"; sequence)
end

function contraction_sequence_to_expr(seq)
    if seq isa AbstractVector
        return prod(contraction_sequence_to_expr, seq)
    else
        return symnameddims(seq)
    end
end

function contraction_sequence(::Algorithm"leftassociative", tn::Vector{<:AbstractArray})
    return contraction_sequence_to_expr(collect(1:length(tn)))
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

function contractnetwork(tn; alg, kwargs...)
    return contractnetwork(set_default_kwargs(Algorithm(alg; kwargs...)), tn)
end
