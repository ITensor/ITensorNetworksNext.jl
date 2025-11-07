using BackendSelection: @Algorithm_str, Algorithm
using Base.Broadcast: materialize
using ITensorNetworksNext.LazyNamedDimsArrays: lazy, optimize_evaluation_order, substitute,
    symnameddims

# This is related to `MatrixAlgebraKit.select_algorithm`.
# TODO: Define this in BackendSelection.jl.
backend_value(::Algorithm{alg}) where {alg} = alg
using BackendSelection: parameters
function merge_parameters(alg::Algorithm; kwargs...)
    return Algorithm(backend_value(alg); merge(parameters(alg), kwargs)...)
end
to_algorithm(alg::Algorithm; kwargs...) = merge_parameters(alg; kwargs...)
to_algorithm(alg; kwargs...) = Algorithm(alg; kwargs...)

# `contract_network`
contract_network(alg::Algorithm, tn) = error("Not implemented.")
function default_kwargs(::typeof(contract_network), tn)
    return (; alg = Algorithm"exact"(; order_alg = Algorithm"eager"()))
end
function contract_network(tn; alg = default_kwargs(contract_network, tn).alg, kwargs...)
    return contract_network(to_algorithm(alg; kwargs...), tn)
end

# `contract_network(::Algorithm"exact", ...)`
function contract_network(alg::Algorithm"exact", tn)
    order = @something begin
        get(alg, :order, nothing)
        contraction_order(
            tn; alg = get(alg, :order_alg, default_kwargs(contraction_order, tn).alg)
        )
    end
    syms_to_ts = Dict(symnameddims(i, Tuple(inds(tn[i]))) => lazy(tn[i]) for i in eachindex(tn))
    tn_expression = substitute(order, syms_to_ts)
    return materialize(tn_expression)
end

# `contraction_order`
function contraction_order end
default_kwargs(::typeof(contraction_order), tn) = (; alg = Algorithm"eager"())
function contraction_order(tn; alg = default_kwargs(contraction_order, tn).alg, kwargs...)
    return contraction_order(to_algorithm(alg; kwargs...), tn)
end
function contraction_order(alg::Algorithm"left_associative", tn)
    return prod(i -> symnameddims(i, Tuple(inds(tn[i]))), eachindex(tn))
end
function contraction_order(alg::Algorithm, tn)
    s = contraction_order(tn; alg = Algorithm"left_associative"())
    return optimize_evaluation_order(s; alg)
end
