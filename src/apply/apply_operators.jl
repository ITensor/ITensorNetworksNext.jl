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

function apply_operator_bp!(
        init, op, iterate;
        cache!, trunc = nothing, pinv_kwargs::NamedTuple = (; tol = 0),
        normalize::Bool = false
    )
    vs = neighbor_vertices(init, op)
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    resolved_envs = isnothing(cache!) ? nothing : boundary_envs(cache!, vs)

    n = length(vs)
    qs = Vector{Any}(undef, n)
    rs = Vector{Any}(undef, n)
    env_invs = Vector{Any}(undef, n)
    r_dimnames = Vector{Any}(undef, n)
    for (i, v) in enumerate(vs)
        ψv = init[v]
        ψv, env_invs[i] = _absorb_envs(ψv, resolved_envs, pinv_kwargs)
        site_v = sitenames(init, v)
        internal_bonds = mapreduce(union, vs; init = eltype(dimnames(ψv))[]) do w
            return if w == v
                eltype(dimnames(ψv))[]
            else
                intersect(dimnames(ψv), dimnames(init[w]))
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
        init[v] = new_ψv
    end
    return init
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
