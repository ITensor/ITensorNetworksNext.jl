import AlgorithmsInterface as AI
import NamedDimsArrays as NDA
using Base: @kwdef
using DataGraphs: AbstractDataGraph
using Graphs: vertices
using LinearAlgebra: norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames
using NamedGraphs.GraphsExtensions: boundary_edges
using TensorAlgebra: TensorAlgebra

function apply_operators(ops, init; op_alg = BPApplyOperator())
    problem = ApplyOperatorsProblem(; operators = ops, init)
    algorithm = ApplyOperators(;
        operator_algorithm = op_alg,
        stopping_criterion = AI.StopAfterIteration(length(ops))
    )
    return AI.solve(problem, algorithm; iterate = copy(init))
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
        algorithm.operator_algorithm, op_i, state.iterate, state.cache
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
        iterate, iteration::Int = 0
    )
    cache = initialize_cache(algorithm.operator_algorithm, iterate)
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return ApplyOperatorsState(;
        iterate, cache, iteration, stopping_criterion_state
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

function apply_operator(
        op,
        init;
        alg = BPApplyOperator(),
        cache = initialize_cache(alg, init)
    )
    return apply_operator(alg, op, init, cache)
end

function apply_operator(alg::BPApplyOperator, op, init, cache)
    return apply_operator_bp(
        op, init, cache;
        trunc = alg.trunc, pinv_kwargs = alg.pinv_kwargs, normalize = alg.normalize
    )
end

function apply_operator_bp(
        op, init, cache;
        trunc = nothing, pinv_kwargs::NamedTuple = (; tol = 0), normalize::Bool = false
    )
    state = copy(init)
    vs = neighbor_vertices(state, op)
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    resolved_envs = isnothing(cache) ? nothing : boundary_envs(cache, vs)

    n = length(vs)
    qs = Vector{Any}(undef, n)
    rs = Vector{Any}(undef, n)
    env_invs = Vector{Any}(undef, n)
    r_dimnames = Vector{Any}(undef, n)
    for (i, v) in enumerate(vs)
        ψv = state[v]
        ψv, env_invs[i] = _absorb_envs(ψv, resolved_envs, pinv_kwargs)
        site_v = sitenames(state, v)
        internal_bonds = mapreduce(union, vs; init = eltype(dimnames(ψv))[]) do w
            return if w == v
                eltype(dimnames(ψv))[]
            else
                intersect(dimnames(ψv), dimnames(state[w]))
            end
        end
        domain = Tuple(union(internal_bonds, site_v))
        codomain = Tuple(setdiff(dimnames(ψv), domain))
        if isempty(codomain)
            qs[i] = nothing
            rs[i] = ψv
        else
            qs[i], rs[i] = TensorAlgebra.qr(ψv, codomain, domain)
        end
        r_dimnames[i] = Set(dimnames(rs[i]))
    end

    blob = NDA.apply(op, reduce(*, rs))

    new_rs = if n == 1
        [blob]
    elseif n == 2
        codomain = Tuple(intersect(dimnames(blob), r_dimnames[1]))
        domain = Tuple(intersect(dimnames(blob), r_dimnames[2]))
        collect(balanced_svd(blob, codomain, domain; trunc))
    else
        throw(ArgumentError("$(n)-site gate decomposition not implemented"))
    end

    for (i, v) in enumerate(vs)
        new_ψv = isnothing(qs[i]) ? new_rs[i] : qs[i] * new_rs[i]
        new_ψv = _absorb_factors(new_ψv, env_invs[i])
        if normalize
            new_ψv = new_ψv / norm(new_ψv)
        end
        state[v] = new_ψv
    end
    return state
end

function neighbor_vertices(tn, op::AbstractNamedDimsArray)
    op_in = domainnames(op)
    return [v for v in vertices(tn) if !isempty(intersect(op_in, sitenames(tn, v)))]
end

function boundary_envs(cache::AbstractDataGraph, vs)
    return [cache[e] for e in boundary_edges(cache, vs; dir = :in)]
end

_absorb_envs(ψ, ::Nothing, _) = (ψ, ())

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
        Y, Yinv = balanced_eigh_and_inv(env, codomain, domain; pinv_kwargs)
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
