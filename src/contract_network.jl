using BackendSelection: @Algorithm_str, Algorithm
using Base.Broadcast: materialize
using NamedDimsArrays: inds
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArrays, Mul, lazy, optimize_evaluation_order,
    substitute, symnameddims

function contract_network(tn; alg = default_kwargs(contract_network, tn).alg)
    return contract_network(alg, tn)
end

contract_network(alg::String, tn) = contract_network(Algorithm(alg), tn)

default_kwargs(::typeof(contract_network), tn) = (; alg = "eager")

function contract_network(
        alg,
        tensors,
    )

    order = contraction_expression(tensors; order = alg)
    symbols_to_tensors = Dict(
        symnameddims(i, tensors[i]) => lazy(tensors[i]) for i in keys(tensors)
    )

    return materialize(substitute(order, symbols_to_tensors))
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
    syms = vec([symnameddims(i, Tuple(inds(tensors[i]))) for i in keys(tensors)])
    return lazy(Mul(syms))
end
function contraction_order(::Algorithm"left_associative", tensors)
    return prod(i -> symnameddims(i, Tuple(inds(tensors[i]))), keys(tensors))
end

function contraction_order(
        order_algorithm::Algorithm,
        tensors,
    )
    order = contraction_order(tensors; order = "flat")
    return optimize_evaluation_order(order; alg = order_algorithm)
end
