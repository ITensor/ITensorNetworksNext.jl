import AlgorithmsInterface as AI
import NamedDimsArrays as NDA
using DataGraphs: AbstractDataGraph
using Graphs: vertices
using LinearAlgebra: norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames
using NamedGraphs.GraphsExtensions: boundary_edges
using TensorAlgebra: TensorAlgebra

function apply_operators(ops, init; op_alg = BPApplyOperator())
    problem = ApplyOperatorsProblem(ops, init)
    algorithm = ApplyOperators(op_alg)
    return AI.solve(problem, algorithm; iterate = copy(init))
end

struct ApplyOperatorsProblem{Ops, Init} <: AI.Problem
    operators::Ops
    init::Init
end

struct ApplyOperators{OpAlg} <: AI.Algorithm
    operator_algorithm::OpAlg
end

mutable struct ApplyOperatorsState{
        Iterate, Cache, SCState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    cache::Cache
    iteration::Int
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

function initialize_cache end

function AI.initialize_state(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators;
        iterate, iteration::Int = 0
    )
    cache = initialize_cache(iterate, algorithm.operator_algorithm)
    sc = AI.StopAfterIteration(length(problem.operators))
    sc_state = AI.initialize_state(problem, algorithm, sc; iterate)
    return ApplyOperatorsState(iterate, cache, iteration, sc_state)
end

function AI.initialize_state!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators,
        state::ApplyOperatorsState; iteration::Int = 0, kwargs...
    )
    state.iteration = iteration
    sc = AI.StopAfterIteration(length(problem.operators))
    AI.initialize_state!(problem, algorithm, sc, state.stopping_criterion_state)
    return state
end

function AI.is_finished!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators,
        state::ApplyOperatorsState
    )
    sc = AI.StopAfterIteration(length(problem.operators))
    return AI.is_finished!(
        problem, algorithm, sc, state.stopping_criterion_state, state
    )
end

struct BPApplyOperator{Trunc, PinvAlg}
    trunc::Trunc
    pinv_alg::PinvAlg
    normalize::Bool
end

function BPApplyOperator(;
        trunc = nothing, pinv_alg = TikhonovPinv(), normalize::Bool = false
    )
    return BPApplyOperator(trunc, pinv_alg, normalize)
end

# TODO: build a fresh `MessageCache` from `iterate` with a sensible default
# initial-message convention (identity / uniform). For now this is a stub that
# returns `nothing`, which makes `apply_operator_bp` fall back to env-free
# simple update.
initialize_cache(iterate, ::BPApplyOperator) = nothing

function apply_operator(op, init; alg = BPApplyOperator(), cache = nothing)
    return apply_operator(alg, op, init, cache)
end

function apply_operator(alg::BPApplyOperator, op, init, cache)
    return apply_operator_bp(
        op, init, cache;
        trunc = alg.trunc, pinv_alg = alg.pinv_alg, normalize = alg.normalize
    )
end

function apply_operator_bp(
        op, init, cache;
        trunc = nothing, pinv_alg = TikhonovPinv(), normalize::Bool = false
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
        ψv, env_invs[i] = _absorb_envs(ψv, resolved_envs, pinv_alg)
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

function _absorb_envs(ψ, envs, pinv_alg)
    inv_factors = []
    for env in envs
        shared = intersect(dimnames(env), dimnames(ψ))
        isempty(shared) && continue
        length(shared) == 1 || error(
            "env must share exactly one dimname with endpoint, got $(length(shared))"
        )
        domain = Tuple(shared)
        codomain = Tuple(setdiff(dimnames(env), shared))
        Y, Yinv = balanced_eigh_and_inv(env, codomain, domain; pinv_alg)
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
