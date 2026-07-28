using Base.Broadcast: materialize
using Base: @kwdef
using ITensorBase: EvaluationOrderAlgorithm, Greedy, Mul, lazy, optimize_evaluation_order,
    substitute, symnameddims

# `contract_network`
@kwdef struct Exact{Order, OrderAlg}
    order::Order = nothing
    order_alg::OrderAlg = Greedy()
end

function contract_network(alg, tn)
    return throw(ArgumentError("`contract_network` algorithm `$(alg)` not implemented."))
end
function contract_network(tn; alg = Exact())
    return contract_network(alg, tn)
end

# `contract_network(::Exact, ...)`
function get_order(alg::Exact, tn)
    # Allow specifying either an explicit `order` or an `order_alg` to compute one.
    order = if !isnothing(alg.order)
        alg.order
    else
        contraction_order(tn; alg = alg.order_alg)
    end
    # Contraction order may or may not have indices attached, canonicalize the format
    # by attaching indices.
    subs = Dict(symnameddims(i) => symnameddims(i, Tuple(axes(t))) for (i, t) in pairs(tn))
    return substitute(order, subs)
end
# Promote the operands to their common type before lowering to the lazy expression, so every lazy
# operand shares one concrete type. Otherwise a network of mixed types (a plain tensor is a trivial
# operator, so mixing operators and plain tensors is the common case) widens the symbolic `Mul`
# container to a `UnionAll` it cannot construct. `promote_type`/`convert` keep an all-plain network
# at the plain type (the promotion is a no-op), so its fast path is unchanged.
function contract_network(alg::Exact, tn)
    order = get_order(alg, tn)
    T = mapreduce(typeof, promote_type, tn)
    syms_to_ts = Dict(
        symnameddims(i, Tuple(axes(t))) => lazy(convert(T, t)) for (i, t) in pairs(tn)
    )
    tn_expression = substitute(order, syms_to_ts)
    return materialize(tn_expression)
end

# `contraction_order`
function contraction_order end
function contraction_order(tn; alg = Greedy())
    return contraction_order(alg, tn)
end
# Convert the tensor network to a flat symbolic multiplication expression.
struct Flat end
function contraction_order(alg::Flat, tn)
    # Same as: `reduce((a, b) -> *(a, b; flatten = true), syms)`.
    syms = vec([symnameddims(i, Tuple(axes(tn[i]))) for i in keys(tn)])
    return lazy(Mul(syms))
end
struct LeftAssociative end
function contraction_order(alg::LeftAssociative, tn)
    return prod(i -> symnameddims(i, Tuple(axes(tn[i]))), keys(tn))
end
# Internal implementation shared with the OMEinsumContractionOrders extension.
function _contraction_order(alg, tn)
    s = contraction_order(Flat(), tn)
    return optimize_evaluation_order(s; alg)
end
function contraction_order(alg::EvaluationOrderAlgorithm, tn)
    return _contraction_order(alg, tn)
end
