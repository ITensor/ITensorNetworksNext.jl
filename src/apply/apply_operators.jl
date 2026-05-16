import AlgorithmsInterface as AI
import NamedDimsArrays as NDA
using Base: @kwdef
using DataGraphs: AbstractDataGraph
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
    vs = neighbor_vertices(init, op)
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
        envs = boundary_envs(cache!, vs)
        ψ_gauge, env_invs = _absorb_envs(ψv, envs, pinv_kwargs)
        ψ_gauge = ψ_gauge / norm(ψ_gauge)
        ψv = _absorb_factors(ψ_gauge, env_invs)
    end
    init[v] = ψv
    return init
end

function apply_operator_bp_nsite!(
        ::Val{2}, init, op, vs;
        cache!, trunc, pinv_kwargs, normalize
    )
    v1, v2 = vs
    envs = boundary_envs(cache!, vs)
    ψ1, env_invs_1 = _absorb_envs(init[v1], envs, pinv_kwargs)
    ψ2, env_invs_2 = _absorb_envs(init[v2], envs, pinv_kwargs)
    bond = Tuple(intersect(dimnames(ψ1), dimnames(ψ2)))
    Q1, R1 = _gate_split(ψ1, sitenames(init, v1), bond)
    Q2, R2 = _gate_split(ψ2, sitenames(init, v2), bond)
    blob = NDA.apply(op, R1 * R2)
    codomain = Tuple(intersect(dimnames(blob), dimnames(R1)))
    domain = Tuple(intersect(dimnames(blob), dimnames(R2)))
    R1_new, R2_new = balanced_svd(blob, codomain, domain; trunc)
    new_ψ1 = Q1 * R1_new
    new_ψ2 = Q2 * R2_new
    new_ψ1 = _absorb_factors(new_ψ1, env_invs_1)
    new_ψ2 = _absorb_factors(new_ψ2, env_invs_2)
    if normalize
        new_ψ1 = new_ψ1 / norm(new_ψ1)
        new_ψ2 = new_ψ2 / norm(new_ψ2)
    end
    init[v1] = new_ψ1
    init[v2] = new_ψ2
    return init
end

function _gate_split(ψ, site, bond)
    domain = Tuple(union(bond, site))
    codomain = Tuple(setdiff(dimnames(ψ), domain))
    return TensorAlgebra.qr(ψ, codomain, domain)
end

function neighbor_vertices(tn, op::AbstractNamedDimsArray)
    op_in = domainnames(op)
    return [v for v in vertices(tn) if !isempty(intersect(op_in, sitenames(tn, v)))]
end

function boundary_envs(cache::AbstractDataGraph, vs)
    return [cache[e] for e in boundary_edges(cache, vs; dir = :in)]
end

function _absorb_envs(ψ, envs, pinv_kwargs)
    inv_factors = []
    for env in envs
        shared = intersect(dimnames(env), dimnames(ψ))
        isempty(shared) && continue
        length(shared) == 1 || error(
            "env must share exactly one dimname with endpoint, got $(length(shared))"
        )
        domain = Tuple(shared)
        codomain = Tuple(setdiff(dimnames(env), shared))
        Y, Yinv = balanced_eigh_and_inv(env, codomain, domain; pinv_kwargs...)
        ψ = ψ * Y
        push!(inv_factors, Yinv)
    end
    return ψ, Tuple(inv_factors)
end

function _absorb_factors(ψ, factors)
    for f in factors
        ψ = ψ * f
    end
    return ψ
end
