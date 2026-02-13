using BackendSelection: @Algorithm_str, Algorithm
using Base.Broadcast: materialize
using NamedDimsArrays: inds
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArrays, Mul, lazy, optimize_evaluation_order,
    substitute, symnameddims

function contract_network(tn; alg = default_kwargs(contract_network, tn).alg)
    return contract_network(alg, tn)
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
    subs = Dict(symnameddims(i) => symnameddims(i, Tuple(axes(tn[i]))) for i in keys(tn))
    return substitute(order, subs)
end
function contract_network(alg::Algorithm"exact", tn)
    order = get_order(alg, tn)
    syms_to_ts = Dict(symnameddims(i, Tuple(axes(tn[i]))) => lazy(tn[i]) for i in keys(tn))
    tn_expression = substitute(order, syms_to_ts)
    return materialize(tn_expression)
end

# `contraction_order`
function contraction_order end
default_kwargs(::typeof(contraction_order), tensors) = (; order = "eager")

function contraction_expression(tensors; order = default_kwargs(contraction_order, tensors).order)
    order = contraction_order(order, tensors)

    # Contraction order may or may not have indices attached, canonicalize the format
    # by attaching indices.
    subs = Dict(symnameddims(i) => symnameddims(i, tensors[i]) for i in keys(tensors))

    return substitute(order, subs)
end

contraction_order(order, tensors) = order
function contraction_order(tensors; order = default_kwargs(contraction_order, tensors).order)
    return contraction_order(Algorithm(order), tensors)
end
# Convert the tensor network to a flat symbolic multiplication expression.
function contraction_order(::Algorithm"flat", tensors)
    # Same as: `reduce((a, b) -> *(a, b; flatten = true), syms)`.
    syms = vec([symnameddims(i, Tuple(axes(tn[i]))) for i in keys(tn)])
    return lazy(Mul(syms))
end
function contraction_order(alg::Algorithm"left_associative", tn)
    return prod(i -> symnameddims(i, Tuple(axes(tn[i]))), keys(tn))
end

function contraction_order(
        order_algorithm::Algorithm,
        tensors,
    )
    order = contraction_order(tensors; order = "flat")
    return optimize_evaluation_order(order; alg = order_algorithm)
end
