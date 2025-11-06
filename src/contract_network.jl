using BackendSelection: @Algorithm_str, Algorithm
using Base.Broadcast: materialize
using ITensorNetworksNext.LazyNamedDimsArrays: lazy, substitute, symnameddims

# This is based on `MatrixAlgebraKit.select_algorithm`.
# TODO: Define this in BackendSelection.jl.
function select_algorithm(alg; kwargs...)
    if alg isa Algorithm
        @assert isempty(kwargs) "Cannot pass keyword arguments when `alg` is an `Algorithm`."
        return alg
    else
        return Algorithm(alg; kwargs...)
    end
end

# `contract_network`
contract_network(alg::Algorithm, tn) = error("Not implemented.")
function default_kwargs(::typeof(contract_network), tn)
    return (; alg = Algorithm"exact"(; order_alg = Algorithm"eager"()))
end
function contract_network(tn; alg = default_kwargs(contract_network, tn).alg, kwargs...)
    return contract_network(select_algorithm(alg; kwargs...), tn)
end

# `contract_network(::Algorithm"exact", ...)`
function contract_network(alg::Algorithm"exact", tn)
    order = @something begin
        get(alg, :order, nothing)
        contraction_order(tn; alg = get(alg, :order_alg, default_kwargs(contraction_order, tn).alg))
    end
    syms_to_ts = Dict(symnameddims(i) => lazy(tn[i]) for i in eachindex(tn))
    tn_expression = substitute(order, syms_to_ts)
    return materialize(tn_expression)
end

# `contraction_order`
contraction_order(alg::Algorithm, tn) = error("Not implemented.")
default_kwargs(::typeof(contraction_order), tn) = (; alg = Algorithm"eager"())
function contraction_order(tn; alg = default_kwargs(contraction_order, tn).alg, kwargs...)
    return contraction_order(select_algorithm(alg; kwargs...), tn)
end

# `contraction_order(::Algorithm"eager", ...)`
function contraction_order(alg::Algorithm"eager", tn)
    return error("Eager not implemented.")
end
