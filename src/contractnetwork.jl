using BackendSelection: @Algorithm_str, Algorithm
using ITensorNetworksNext.LazyNamedDimsArrays: substitute, materialize, lazy,
    symnameddims

#Algorithmic defaults
default_sequence_alg(::Algorithm"exact") = "leftassociative"
default_sequence(::Algorithm"exact") = nothing
function set_default_kwargs(alg::Algorithm"exact")
    sequence = get(alg, :sequence, nothing)
    sequence_alg = get(alg, :sequence_alg, default_sequence_alg(alg))
    return Algorithm("exact"; sequence, sequence_alg)
end

function contraction_sequence_to_expr(seq)
    if seq isa AbstractVector
        return prod(contraction_sequence_to_expr, seq)
    else
        return symnameddims(seq)
    end
end

function contraction_sequence(::Algorithm"leftassociative", tn::Vector{<:AbstractArray})
    return prod(symnameddims, 1:length(tn))
end

function contraction_sequence(tn::Vector{<:AbstractArray}; sequence_alg = default_sequence_alg(Algorithm("exact")))
    return contraction_sequence(Algorithm(sequence_alg), tn)
end

function contractnetwork(alg::Algorithm"exact", tn::Vector{<:AbstractArray})
    if !isnothing(alg.sequence)
        sequence = alg.sequence
    else
        sequence = contraction_sequence(tn; sequence_alg = alg.sequence_alg)
    end

    sequence = substitute(sequence, Dict(symnameddims(i) => lazy(tn[i]) for i in 1:length(tn)))
    return materialize(sequence)
end

function contractnetwork(alg::Algorithm"exact", tn::AbstractTensorNetwork)
    return contractnetwork(alg, [tn[v] for v in vertices(tn)])
end

function contractnetwork(tn; alg, kwargs...)
    return contractnetwork(set_default_kwargs(Algorithm(alg; kwargs...)), tn)
end
