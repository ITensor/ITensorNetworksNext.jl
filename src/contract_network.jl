using BackendSelection: @Algorithm_str, Algorithm
using Base.Broadcast: materialize
using ITensorNetworksNext.LazyNamedDimsArrays: Mul, lazy, optimize_evaluation_order,
    substitute, symnameddims

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
function contract_network(alg::Algorithm, tn)
    return throw(ArgumentError("`contract_network` algorithm `$(alg)` not implemented."))
end
function default_kwargs(::typeof(contract_network), tn)
    return (; alg = Algorithm"exact"(; order_alg = Algorithm"eager"()))
end
function contract_network(tn; alg = default_kwargs(contract_network, tn).alg, kwargs...)
    return contract_network(to_algorithm(alg; kwargs...), tn)
end

# `contract_network(::Algorithm"exact", ...)`
function get_order(alg::Algorithm"exact", tn)
    # Allow specifying either `order` or `order_alg`.
    order = get(alg, :order, nothing)
    order = if !isnothing(order)
        order
    else
        default_order_alg = default_kwargs(contraction_order, tn).alg
        order_alg = get(alg, :order_alg, default_order_alg)
        # TODO: Capture other keyword arguments and pass them to `contraction_order`.
        contraction_order(tn; alg = order_alg)
    end
    # Contraction order may or may not have indices attached, canonicalize the format
    # by attaching indices.
    subs = Dict(symnameddims(i) => symnameddims(i, Tuple(inds(tn[i]))) for i in keys(tn))
    return substitute(order, subs)
end
function contract_network(alg::Algorithm"exact", tn)
    order = get_order(alg, tn)
    syms_to_ts = Dict(symnameddims(i, Tuple(inds(tn[i]))) => lazy(tn[i]) for i in keys(tn))
    tn_expression = substitute(order, syms_to_ts)
    return materialize(tn_expression)
end

# `contraction_order`
function contraction_order end
default_kwargs(::typeof(contraction_order), tn) = (; alg = Algorithm"eager"())
function contraction_order(tn; alg = default_kwargs(contraction_order, tn).alg, kwargs...)
    return contraction_order(to_algorithm(alg; kwargs...), tn)
end
# Convert the tensor network to a flat symbolic multiplication expression.
function contraction_order(alg::Algorithm"flat", tn)
    # Same as: `reduce((a, b) -> *(a, b; flatten = true), syms)`.
    syms = vec([symnameddims(i, Tuple(inds(tn[i]))) for i in keys(tn)])
    return lazy(Mul(syms))
end
function contraction_order(alg::Algorithm"left_associative", tn)
    return prod(i -> symnameddims(i, Tuple(inds(tn[i]))), keys(tn))
end
function contraction_order(alg::Algorithm, tn)
    s = contraction_order(Algorithm"flat"(), tn)
    return optimize_evaluation_order(s; alg)
end
