import AlgorithmsInterface as AI
import MatrixAlgebraKit as MAK
import NamedDimsArrays as NDA
import TensorAlgebra as TA
using Base: @kwdef
using Graphs: dst, src, vertices
using LinearAlgebra: norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames, nameddims, randname,
    replacedimnames, state
using NamedGraphs.GraphsExtensions: all_edges, boundary_edges

# === NestedAlgorithm framework ===

abstract type NestedAlgorithm <: AI.Algorithm end

function initialize_subsolve(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State
    )
    return throw(MethodError(initialize_subsolve, (problem, algorithm, state)))
end

function finalize_substate!(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State, substate::AI.State
    )
    state.iterate = substate.iterate
    return state
end

function AI.step!(problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State)
    subproblem, subalgorithm, substate = initialize_subsolve(problem, algorithm, state)
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

function initialize_subsolve(
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

# Initialize the BP message cache to identity square-root messages.
function initialize_cache(
        ::ApplyOperatorProblem, ::BPApplyGate, iterate::AbstractTensorNetwork
    )
    return sqrtmessagecache(all_edges(iterate)) do edge
        factor = iterate[dst(edge)]
        return state(one(similar_operator(factor, linkaxes(iterate, edge))))
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

    # qr codomain at v_i: legs of ψ_v_i not shared with ψ_v_j (the v1v2 bond)
    # and not touched by `op` (those need to stay in `R` so the gate can act
    # on them). `setdiff(_, dimnames(op))` is safe even though `op` carries
    # legs not in ψ_v_i — extra elements in the subtracted set are no-ops.
    Q_v1, R_v1 = TA.qr(ψ_v1, setdiff(dimnames(ψ_v1), dimnames(ψ_v2), dimnames(op)))
    Q_v2, R_v2 = TA.qr(ψ_v2, setdiff(dimnames(ψ_v2), dimnames(ψ_v1), dimnames(op)))
    op_R_v1v2 = NDA.apply(op, R_v1 * R_v2)
    # `op_R_v1v2 ≈ U_v1 · S · U_v2`. Absorb `√S` symmetrically into the
    # new `R_v1`, `R_v2` ("balanced gauge"); the same `√S` factor becomes
    # the sqrt-message written back to `cache!` below.
    U_v1, S, U_v2 = TA.svd(op_R_v1v2, setdiff(dimnames(R_v1), dimnames(R_v2)); trunc)
    if normalize
        S = S / norm(S)
    end
    name_v1, name_v2 = dimnames(S)
    # `sqrt(S, (name_v1,), (name_v2,))` is NDA's matrix sqrt of `S` —
    # a single 2-leg named array with dimnames `(name_v1, name_v2)`
    # satisfying `sqrt_S * sqrt_S ≈ S` in the matrix algebra (each
    # `sqrt_S` factor contracts on one of `S`'s legs). Eventual endpoint:
    # 1-arg `sqrt(S)` once `TA.svd` returns `S` as a `NamedDimsOperator`.
    sqrt_S = sqrt(S, (name_v1,), (name_v2,))
    # Build R factors by absorbing `sqrt_S` on each side; the rebind on
    # the v1 side picks `name_v1` as the new shared bond between
    # `dest[v1]` and `dest[v2]`. With a `NamedDimsOperator` wrapper, the
    # rebind becomes `apply(sqrt_S, U_v1)`.
    R_v1 = replacedimnames(U_v1 * sqrt_S, name_v2 => name_v1)
    R_v2 = sqrt_S * U_v2

    dest[v1] = prod([[Q_v1 * R_v1]; inv_sqrt_envs_v1])
    dest[v2] = prod([[Q_v2 * R_v2]; inv_sqrt_envs_v2])

    # Both directed sqrt-messages derive from the same `sqrt_S`, but
    # with different name-slot choices so each message's "matching" leg
    # (name_v1, contracting with the receiving tensor) carries the
    # correct arrow direction.
    #
    # `dest[v1]`'s name_v1 bond inherits the domain-side arrow of `S`
    # (from the `name_v2 => name_v1` rebind in `R_v1`), and `dest[v2]`'s
    # name_v1 bond inherits the codomain-side arrow (from `sqrt_S * U_v2`).
    # So:
    #   * `cache![v2 => v1]`'s matching leg needs the codomain-side arrow
    #     → use sqrt_S's name_v1 leg directly; relabel name_v2 to fresh.
    #   * `cache![v1 => v2]`'s matching leg needs the domain-side arrow
    #     → swap roles: rename sqrt_S's name_v2 to name_v1, and the
    #     original name_v1 (now the internal-rank slot) to a fresh name.
    # For dense backings sqrt_S equals its transpose, so the two choices
    # coincide numerically; the distinction matters for graded /
    # fermionic axes.
    cache![v1 => v2] = replacedimnames(
        sqrt_S, name_v1 => randname(name_v1), name_v2 => name_v1
    )
    cache![v2 => v1] = replacedimnames(sqrt_S, name_v2 => randname(name_v2))
    return dest
end
