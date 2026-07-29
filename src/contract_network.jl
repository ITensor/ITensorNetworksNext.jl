using Base.Broadcast: materialize
using Base: @kwdef
using ITensorBase: EvaluationOrderAlgorithm, Greedy, Mul, ismul, lazy,
    optimize_evaluation_order, substitute, symnameddims, to_mul_arguments

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
# The contraction leaves of an operand: a lazy product (the `NormNetwork` doubled vertex
# `lazy(ket) * lazy(conj(bra))`) contributes each factor, recursively; anything else is one leaf.
leaf_tensors(t) = ismul(t) ? mapreduce(leaf_tensors, vcat, to_mul_arguments(t)) : [t]

# Promote the operands to their common type before lowering to the lazy expression, so every lazy
# operand shares one concrete type. Otherwise a network of mixed types (a plain tensor is a trivial
# operator, so mixing operators and plain tensors is the common case) widens the symbolic `Mul`
# container to a `UnionAll` it cannot construct. `promote_type`/`convert` keep an all-plain network
# at the plain type (the promotion is a no-op), so its fast path is unchanged.
#
# For a computed order, expand each promoted operand into its contraction leaves so the order
# optimizer sees each factor of a lazy product — and the physical index shared between the ket and
# bra of a doubled vertex — instead of one opaque node with only the outer bond legs. Without this
# the optimizer cannot interleave the other operands between the two layers and is forced to form
# the doubled `ket * conj(bra)` tensor first (χ^(2·degree)). Flattening the *promoted* operand keeps
# the operator/state semantics the promotion just established. An explicit operand-level `order` is
# honored over the operands as given, so it is not flattened.
function contract_network(alg::Exact, tn)
    if !isnothing(alg.order)
        # Explicit order: honor it over the operands as given, so it is not flattened.
        order = get_order(alg, tn)
        T = mapreduce(typeof, promote_type, tn)
        syms_to_ts = Dict(
            symnameddims(i, Tuple(axes(t))) => lazy(convert(T, t)) for (i, t) in pairs(tn)
        )
        return materialize(substitute(order, syms_to_ts))
    end
    # Computed order: expand each promoted operand into its contraction leaves.
    T = mapreduce(typeof, promote_type, tn)
    leaves = collect(Iterators.flatten(leaf_tensors(lazy(convert(T, t))) for t in tn))
    order = get_order(alg, leaves)
    syms_to_ts =
        Dict(symnameddims(i, Tuple(axes(t))) => lazy(t) for (i, t) in pairs(leaves))
    return materialize(substitute(order, syms_to_ts))
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
