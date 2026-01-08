using NamedDimsArrays: denamed, inds
using TermInterface: arguments, arity, operation

# The time complexity of evaluating `f(args...)`.
function time_complexity(f, args...)
    return error("Not implemented.")
end
# The space complexity of evaluating `f(args...)`.
function space_complexity(f, args...)
    return error("Not implemented.")
end
# The space complexity of `args`.
function input_space_complexity(f, args...)
    return error("Not implemented.")
end

using NamedDimsArrays: AbstractNamedDimsArray
function time_complexity(
        ::typeof(*), t1::AbstractNamedDimsArray, t2::AbstractNamedDimsArray
    )
    return prod(length ∘ denamed, (inds(t1) ∪ inds(t2)))
end
function time_complexity(
        ::typeof(+), t1::AbstractNamedDimsArray, t2::AbstractNamedDimsArray
    )
    @assert issetequal(inds(t1), inds(t2))
    return prod(denamed, size(t1))
end
function time_complexity(::typeof(*), c::Number, t::AbstractNamedDimsArray)
    return prod(denamed, size(t))
end
function time_complexity(::typeof(*), t::AbstractNamedDimsArray, c::Number)
    return time_complexity(*, c, t)
end

function evaluation_time_complexity(a)
    t = Ref(0)
    opwalk(a) do f
        return function (args...)
            t[] += time_complexity(f, args...)
            return f(args...)
        end
    end
    return t[]
end

# The workspace complexity of evaluating expression.
function evaluation_space_complexity(a)
    # TODO: Walk the expression and call `space_complexity` on each node.
    return error("Not implemented.")
end
# The complexity of storing the arguments of the expression.
function argument_space_complexity(a)
    # TODO: Walk the expression and call `input_space_complexity` on each node.
    return error("Not implemented.")
end

# Flatten a nested expression down to a flat expression,
# removing information about the order of operations.
function flatten_expression(a)
    if !iscall(a)
        return a
    elseif ismul(a)
        flattened_arguments = mapreduce(to_mul_arguments, vcat, arguments(a))
        return lazy(Mul(flattened_arguments))
    else
        return error("Variant not supported.")
    end
end

function optimize_evaluation_order(alg, a)
    if !iscall(a)
        return a
    elseif ismul(a)
        return optimize_contraction_order(alg, a)
    else
        # TODO: Recurse into other operations, calling `optimize_evaluation_order`.
        return error("Variant not supported.")
    end
end

function optimize_evaluation_order(
        a; alg = default_optimize_evaluation_order_alg(a)
    )
    return optimize_evaluation_order(alg, a)
end

using BackendSelection: @Algorithm_str, Algorithm
default_optimize_evaluation_order_alg(a) = Algorithm"eager"()

function optimize_contraction_order(alg, a)
    return error("`alg = $alg` not supported.")
end

using Combinatorics: combinations
function optimize_contraction_order(alg::Algorithm"eager", a)
    @assert ismul(a)
    arity(a) in (1, 2) && return a
    a1, a2 = argmin(combinations(arguments(a), 2)) do (a1, a2)
        # Penalize outer product contractions.
        # TODO: Still order the outer products by time complexity,
        # say by checking if there are only outer products left.
        isdisjoint(inds(a1), inds(a2)) && return typemax(Int)
        return time_complexity(*, a1, a2)
    end
    contracted_arguments = [filter(∉((a1, a2)), arguments(a)); [a1 * a2]]
    return optimize_contraction_order(alg, lazy(Mul(contracted_arguments)))
end
