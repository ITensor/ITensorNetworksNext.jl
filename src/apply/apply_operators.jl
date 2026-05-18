import AlgorithmsInterface as AI
import MatrixAlgebraKit as MAK
import NamedDimsArrays as NDA
import TensorAlgebra as TA
using Base: @kwdef
using Graphs: dst, src, vertices
using LinearAlgebra: I, diag, diagm, norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames, nameddims, randname
using NamedGraphs.GraphsExtensions: all_edges, boundary_edges

# === NestedAlgorithm framework ===

abstract type NestedAlgorithm <: AI.Algorithm end

function initialize_subproblem(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State
    )
    return throw(MethodError(initialize_subproblem, (problem, algorithm, state)))
end

function finalize_substate!(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State, substate::AI.State
    )
    state.iterate = substate.iterate
    return state
end

function AI.step!(problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State)
    subproblem, subalgorithm, substate = initialize_subproblem(problem, algorithm, state)
    AI.solve!(subproblem, subalgorithm, substate)
    finalize_substate!(problem, algorithm, state, substate)
    return state
end

# === apply_operators (plural, iterative over a list of operators) ===

function apply_operators(ops, state; op_alg = BPApplyGate(), kwargs...)
    problem = ApplyOperatorsProblem(; operators = ops, init = state)
    algorithm = ApplyOperators(;
        operator_algorithm = op_alg,
        stopping_criterion = AI.StopAfterIteration(length(ops))
    )
    return AI.solve(problem, algorithm; iterate = copy(state), kwargs...)
end

@kwdef struct ApplyOperatorsProblem{Ops, Init} <: AI.Problem
    operators::Ops
    init::Init
end

@kwdef struct ApplyOperators{OpAlg} <: NestedAlgorithm
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

function AI.initialize_state(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators;
        iterate,
        cache! = initialize_cache(problem, algorithm, iterate),
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

function initialize_subproblem(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators,
        state::ApplyOperatorsState
    )
    op_i = problem.operators[state.iteration]
    subproblem = ApplyOperatorProblem(; op = op_i, init = state.iterate)
    subalgorithm = algorithm.operator_algorithm
    substate = AI.initialize_state(
        subproblem, subalgorithm; state.iterate, cache! = state.cache
    )
    return subproblem, subalgorithm, substate
end

function initialize_cache(problem::AI.Problem, algorithm::AI.Algorithm, iterate)
    return throw(MethodError(initialize_cache, (problem, algorithm, iterate)))
end

function initialize_cache(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators, iterate
    )
    subproblem = ApplyOperatorProblem(; op = first(problem.operators), init = iterate)
    subalgorithm = algorithm.operator_algorithm
    return initialize_cache(subproblem, subalgorithm, iterate)
end

# === apply_operator (singular, one gate application) ===

@kwdef struct ApplyOperatorProblem{Op, Init} <: AI.Problem
    op::Op
    init::Init
end

function apply_operator(op, state; alg = BPApplyGate(), kwargs...)
    problem = ApplyOperatorProblem(; op, init = state)
    return AI.solve(problem, alg; iterate = copy(state), kwargs...)
end

function apply_operator!(dest, op, state; alg = BPApplyGate(), kwargs...)
    problem = ApplyOperatorProblem(; op, init = state)
    alg_state = AI.initialize_state(problem, alg; iterate = dest, kwargs...)
    return AI.solve!(problem, alg, alg_state)
end

# === BPApplyGate (non-iterative; overloads solve_loop! directly) ===

@kwdef struct BPApplyGate{Trunc, PinvKwargs <: NamedTuple} <: AI.Algorithm
    trunc::Trunc = nothing
    pinv_kwargs::PinvKwargs = (; tol = 0)
    normalize::Bool = false
end

@kwdef mutable struct BPApplyGateState{Iterate, Cache} <: AI.State
    iterate::Iterate
    cache::Cache
end

function AI.initialize_state(
        problem::ApplyOperatorProblem, algorithm::BPApplyGate;
        iterate, cache! = initialize_cache(problem, algorithm, iterate)
    )
    return BPApplyGateState(; iterate, cache = cache!)
end

# Non-iterative algorithm: no per-call state to reset.
function AI.initialize_state!(
        ::ApplyOperatorProblem, ::BPApplyGate, state::BPApplyGateState
    )
    return state
end

# Identity-message cache: trivial Vidal-gauge initialization where each bond
# carries the identity 2-leg matrix (= √I = I, in sqrt-message form). Stored
# in a `SqrtMessageCache` so the BP simple update knows to use the messages
# as gauge-in factors directly and skip the √ step.
function initialize_cache(
        problem::ApplyOperatorProblem, ::BPApplyGate, iterate::AbstractTensorNetwork
    )
    T = eltype(iterate[first(vertices(iterate))])
    return sqrtmessagecache(all_edges(iterate)) do edge
        bond_name = only(linknames(iterate, edge))
        n = Int(length(only(linkaxes(iterate, edge))))
        fresh_name = randname(bond_name)
        # TODO: Make this work for symmetric tensors (GradedArrays): construct
        # an identity that respects the sector structure of the bond axis,
        # rather than a plain `Matrix{T}(I, n, n)` keyed only by length.
        return nameddims(Matrix{T}(I, n, n), (fresh_name, bond_name))
    end
end

# Non-iterative algorithm: bypass the step!/stopping-criterion loop.
function AI.solve_loop!(
        problem::ApplyOperatorProblem, algorithm::BPApplyGate,
        state::BPApplyGateState
    )
    apply_gate_bp!(
        state.iterate, problem.op, problem.init;
        cache! = state.cache,
        trunc = algorithm.trunc, pinv_kwargs = algorithm.pinv_kwargs,
        normalize = algorithm.normalize
    )
    return state
end

# === BP simple-update implementation ===
#
# The `cache!` here is assumed to be a `SqrtMessageCache`: messages on each
# directed edge are sqrt-form (√M), so they are used as gauge-in factors
# directly and only the (regularized) inverse is needed for gauge-out.

function apply_gate_bp!(
        dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork; kwargs...
    )
    op_in = domainnames(op)
    vs = [v for v in vertices(state) if !isempty(intersect(op_in, sitenames(state, v)))]
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    return apply_gate_bp_nsite!(Val(length(vs)), dest, op, state, vs; kwargs...)
end

function apply_gate_bp_nsite!(
        ::Val{N}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs; kwargs...
    ) where {N}
    return throw(ArgumentError("$N-site gate decomposition not implemented"))
end

function apply_gate_bp_nsite!(
        ::Val{1}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs;
        cache!, normalize, kwargs...
    )
    v = only(vs)
    ψv = NDA.apply(op, state[v])
    if normalize
        sqrt_envs = [cache![e] for e in boundary_edges(cache!, vs; dir = :in)]
        ψv /= norm(prod([[ψv]; sqrt_envs]))
    end
    dest[v] = ψv
    return dest
end

function apply_gate_bp_nsite!(
        ::Val{2}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs;
        cache!, trunc, pinv_kwargs, normalize
    )
    v1, v2 = vs
    edges_in = boundary_edges(cache!, vs; dir = :in)
    sqrt_envs_v1 = [cache![e] for e in edges_in if dst(e) == v1]
    sqrt_envs_v2 = [cache![e] for e in edges_in if dst(e) == v2]
    inv_sqrt_envs_v1 = map(sqrt_envs_v1) do env
        return MAK.inv_regularized(
            env, setdiff(dimnames(env), dimnames(state[v1])); pinv_kwargs...
        )
    end
    inv_sqrt_envs_v2 = map(sqrt_envs_v2) do env
        return MAK.inv_regularized(
            env, setdiff(dimnames(env), dimnames(state[v2])); pinv_kwargs...
        )
    end

    ψ_v1 = prod([[state[v1]]; sqrt_envs_v1])
    ψ_v2 = prod([[state[v2]]; sqrt_envs_v2])

    # Site legs of `op` at v1 / v2 — `intersect` rather than
    # `sitenames(state, v_i)` so we only put the *actually-acted-on* site
    # legs into the qr domain (the gate may touch a strict subset).
    s_v1 = intersect(dimnames.((ψ_v1, op))...)
    s_v2 = intersect(dimnames.((ψ_v2, op))...)
    Q_v1, R_v1 = TA.qr(ψ_v1, setdiff(dimnames.((ψ_v1, ψ_v2))..., s_v1))
    Q_v2, R_v2 = TA.qr(ψ_v2, setdiff(dimnames.((ψ_v2, ψ_v1))..., s_v2))
    op_R_v1v2 = NDA.apply(op, R_v1 * R_v2)
    # `op_R_v1v2 ≈ U · S · V`, with `S` a 2-leg diagonal NamedDimsArray
    # on `(name_u, name_v)`. Absorb `√S` symmetrically into the new
    # `R_v1`, `R_v2` ("balanced gauge") and unify the two SVD bond names
    # into a single fresh `new_bond` so the gauged tensors share one
    # bond; the same `√σ` becomes the sqrt-message written back to
    # `cache!` below.
    U, S, V = TA.svd(
        op_R_v1v2,
        intersect(dimnames(op_R_v1v2), dimnames(R_v1)),
        intersect(dimnames(op_R_v1v2), dimnames(R_v2));
        trunc
    )
    name_u, name_v = dimnames(S)
    sqrtσ = sqrt.(diag(S.denamed))
    new_bond = randname(name_u)
    sqrt_S_left = nameddims(diagm(sqrtσ), (name_u, new_bond))
    sqrt_S_right = nameddims(diagm(sqrtσ), (new_bond, name_v))
    R_v1 = U * sqrt_S_left
    R_v2 = sqrt_S_right * V

    ψ_v1 = prod([[Q_v1 * R_v1]; inv_sqrt_envs_v1])
    ψ_v2 = prod([[Q_v2 * R_v2]; inv_sqrt_envs_v2])
    if normalize
        ψ_v1 = ψ_v1 / norm(ψ_v1)
        ψ_v2 = ψ_v2 / norm(ψ_v2)
    end
    dest[v1] = ψ_v1
    dest[v2] = ψ_v2

    # Write fresh sqrt-messages on the (v1, v2) edge of the cache, so that the
    # cache stays consistent with the new bond name and weights in `dest`.
    W = diagm(sqrtσ)
    cache![v1 => v2] = nameddims(W, (randname(new_bond), new_bond))
    cache![v2 => v1] = nameddims(W, (randname(new_bond), new_bond))
    return dest
end
