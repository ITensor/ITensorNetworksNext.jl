import AlgorithmsInterface as AI
import NamedDimsArrays as NDA
using Base: @kwdef
using Graphs: vertices
using LinearAlgebra: norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames
using NamedGraphs.GraphsExtensions: boundary_edges
using TensorAlgebra: TensorAlgebra

function apply_operators(
        ops, iterate;
        op_alg = BPApplyOperator(), cache! = initialize_cache(op_alg, iterate)
    )
    problem = ApplyOperatorsProblem(; operators = ops, init = iterate)
    algorithm = ApplyOperators(;
        operator_algorithm = op_alg,
        stopping_criterion = AI.StopAfterIteration(length(ops))
    )
    return AI.solve(problem, algorithm; iterate, cache!)
end

@kwdef struct ApplyOperatorsProblem{Ops, Init} <: AI.Problem
    operators::Ops
    init::Init
end

@kwdef struct ApplyOperators{OpAlg} <: AI.Algorithm
    operator_algorithm::OpAlg
    stopping_criterion::AI.StopAfterIteration
end

@kwdef mutable struct ApplyOperatorsState{
        Iterate, Cache, SCState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    cache::Cache
    iteration::Int = 0
    stopping_criterion_state::SCState
end

function AI.step!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators,
        state::ApplyOperatorsState
    )
    op_i = problem.operators[state.iteration]
    state.iterate = apply_operator(
        algorithm.operator_algorithm, op_i, state.iterate;
        (cache!) = state.cache
    )
    return state
end

"""
    initialize_cache(algorithm, iterate)

Construct the cache stored on [`ApplyOperatorsState`](@ref) for the per-operator
`algorithm` (e.g. [`BPApplyOperator`](@ref)) given the initial `iterate`.
Throws a `MethodError` by default; per-algorithm methods opt in.
"""
function initialize_cache(algorithm, iterate)
    return throw(MethodError(initialize_cache, (algorithm, iterate)))
end

function AI.initialize_state(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators;
        iterate, cache! = initialize_cache(algorithm.operator_algorithm, iterate),
        iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return ApplyOperatorsState(;
        iterate, cache = cache!, iteration, stopping_criterion_state
    )
end

function AI.initialize_state!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators,
        state::ApplyOperatorsState; iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion,
        state.stopping_criterion_state
    )
    return state
end

@kwdef struct BPApplyOperator{Trunc, PinvKwargs <: NamedTuple}
    trunc::Trunc = nothing
    pinv_kwargs::PinvKwargs = (; tol = 0)
    normalize::Bool = false
end

"""
    initialize_output(::typeof(apply_operator), algorithm, op, iterate)

Allocate the output buffer that [`apply_operator!`](@ref) writes into. The
default uses `copy(iterate)` as the starting guess; per-algorithm methods
may override.
"""
initialize_output(::typeof(apply_operator), algorithm, op, iterate) = copy(iterate)

"""
    apply_operator(op, iterate; alg, cache!)
    apply_operator(algorithm, op, iterate; cache!)

Apply the operator `op` to the input tensor network `iterate` under
`algorithm`, returning the new tensor network. The cache `cache!` is mutated
in place (the `!` suffix marks it as a mutated kwarg).
"""
function apply_operator(
        op, iterate;
        alg = BPApplyOperator(), cache! = initialize_cache(alg, iterate)
    )
    return apply_operator(alg, op, iterate; cache!)
end

function apply_operator(
        algorithm, op, iterate;
        cache! = initialize_cache(algorithm, iterate)
    )
    init = initialize_output(apply_operator, algorithm, op, iterate)
    apply_operator!(algorithm, init, op, iterate; cache!)
    return init
end

"""
    apply_operator!(algorithm, init, op, iterate; cache!)

In-place form of [`apply_operator`](@ref): writes the result into `init` and
mutates `cache!`. Returns `init`. Throws a `MethodError` by default;
per-algorithm methods opt in.
"""
function apply_operator!(algorithm, init, op, iterate; cache!)
    return throw(MethodError(apply_operator!, (algorithm, init, op, iterate)))
end

function apply_operator!(alg::BPApplyOperator, init, op, iterate; cache!)
    return apply_operator_bp!(
        init, op, iterate;
        cache!, trunc = alg.trunc, pinv_kwargs = alg.pinv_kwargs,
        normalize = alg.normalize
    )
end

function apply_operator_bp!(init, op, iterate; kwargs...)
    op_in = domainnames(op)
    vs = [v for v in vertices(init) if !isempty(intersect(op_in, sitenames(init, v)))]
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    return apply_operator_bp_nsite!(Val(length(vs)), init, op, vs; kwargs...)
end

function apply_operator_bp_nsite!(::Val{N}, init, op, vs; kwargs...) where {N}
    throw(ArgumentError("$N-site gate decomposition not implemented"))
end

function apply_operator_bp_nsite!(
        ::Val{1}, init, op, vs;
        cache!, pinv_kwargs, normalize, kwargs...
    )
    v = only(vs)
    ψv = NDA.apply(op, init[v])
    if normalize
        envs = [cache![e] for e in boundary_edges(cache!, vs; dir = :in)]
        envs_v = filter(e -> !isempty(intersect(dimnames(e), dimnames(init[v]))), envs)
        sqrt_envs_and_invs = map(envs_v) do env
            shared = intersect(dimnames(env), dimnames(init[v]))
            return balanced_eigh_and_inv(
                env, Tuple(setdiff(dimnames(env), shared)), Tuple(shared);
                pinv_kwargs...
            )
        end
        sqrt_envs, inv_sqrt_envs = first.(sqrt_envs_and_invs), last.(sqrt_envs_and_invs)
        ψ_gauge = prod([ψv; sqrt_envs])
        ψv = prod([ψ_gauge / norm(ψ_gauge); inv_sqrt_envs])
    end
    init[v] = ψv
    return init
end

function apply_operator_bp_nsite!(
        ::Val{2}, init, op, vs;
        cache!, trunc, pinv_kwargs, normalize
    )
    v1, v2 = vs
    envs = [cache![e] for e in boundary_edges(cache!, vs; dir = :in)]
    envs_v1 = filter(e -> !isempty(intersect(dimnames(e), dimnames(init[v1]))), envs)
    envs_v2 = filter(e -> !isempty(intersect(dimnames(e), dimnames(init[v2]))), envs)
    sqrt_envs_and_invs_v1 = map(envs_v1) do env
        shared = intersect(dimnames(env), dimnames(init[v1]))
        return balanced_eigh_and_inv(
            env, Tuple(setdiff(dimnames(env), shared)), Tuple(shared); pinv_kwargs...
        )
    end
    sqrt_envs_and_invs_v2 = map(envs_v2) do env
        shared = intersect(dimnames(env), dimnames(init[v2]))
        return balanced_eigh_and_inv(
            env, Tuple(setdiff(dimnames(env), shared)), Tuple(shared); pinv_kwargs...
        )
    end
    sqrt_envs_v1, inv_sqrt_envs_v1 =
        first.(sqrt_envs_and_invs_v1), last.(sqrt_envs_and_invs_v1)
    sqrt_envs_v2, inv_sqrt_envs_v2 =
        first.(sqrt_envs_and_invs_v2), last.(sqrt_envs_and_invs_v2)

    ψ_v1 = prod([init[v1]; sqrt_envs_v1])
    ψ_v2 = prod([init[v2]; sqrt_envs_v2])

    s_v1 = sitenames(init, v1)
    s_v2 = sitenames(init, v2)
    bond = Tuple(intersect(dimnames(ψ_v1), dimnames(ψ_v2)))
    Q_v1, R_v1 = TensorAlgebra.qr(
        ψ_v1, Tuple(setdiff(dimnames(ψ_v1), bond, s_v1)), (bond..., s_v1...)
    )
    Q_v2, R_v2 = TensorAlgebra.qr(
        ψ_v2, Tuple(setdiff(dimnames(ψ_v2), bond, s_v2)), (bond..., s_v2...)
    )
    blob = NDA.apply(op, R_v1 * R_v2)
    R_v1, R_v2 = balanced_svd(
        blob,
        Tuple(intersect(dimnames(blob), dimnames(R_v1))),
        Tuple(intersect(dimnames(blob), dimnames(R_v2)));
        trunc
    )

    ψ_v1 = prod([Q_v1 * R_v1; inv_sqrt_envs_v1])
    ψ_v2 = prod([Q_v2 * R_v2; inv_sqrt_envs_v2])
    if normalize
        ψ_v1 = ψ_v1 / norm(ψ_v1)
        ψ_v2 = ψ_v2 / norm(ψ_v2)
    end
    init[v1] = ψ_v1
    init[v2] = ψ_v2
    return init
end
