import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
import NamedDimsArrays as NDA
import TensorAlgebra as TA
using Base: @kwdef
using Graphs: dst, src, vertices
using LinearAlgebra: norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames, nameddims, operator,
    randname, replacedimnames
using NamedGraphs.GraphsExtensions: all_edges, boundary_edges

# === Top-level user entry point ===

function apply_operators(operators, state, env; alg = nothing, kwargs...)
    algorithm = select_algorithm(
        apply_operators, alg, (operators, state, env); kwargs...
    )
    return apply_operators(algorithm, operators, state, env)
end

# The `apply_operators` iteration algorithm wraps the per-operator algorithm,
# which is itself resolved via `apply_operator` (overridable with `operator_alg`).
function default_algorithm(
        ::typeof(apply_operators), ::Type{Args};
        operator_alg = nothing, environment_alg = nothing, kwargs...
    ) where {Args <: Tuple}
    # `apply_operator` acts on a single operator, so select on the operator
    # element type, keeping the remaining `(state, env)` argument types.
    operators_type, rest... = fieldtypes(Args)
    operator_args = Tuple{eltype(operators_type), rest...}
    operator_algorithm =
        select_algorithm(apply_operator, operator_alg, operator_args; kwargs...)
    environment_algorithm = select_algorithm(prepare_environment, environment_alg, Args)
    return ApplyOperatorsAlgorithm(; operator_algorithm, environment_algorithm)
end

function apply_operators(algorithm, operators, state, env)
    problem = ApplyOperatorsProblem(; operators, init = state)
    # One step per operator. `select_algorithm` dispatches on argument *types*,
    # so `length(operators)` can't reach it; the operator-count bound is set here,
    # where the value is available.
    iteration_algorithm = ApplyOperatorsAlgorithm(;
        algorithm.operator_algorithm,
        algorithm.environment_algorithm,
        stopping_criterion = AI.StopAfterIteration(length(operators))
    )
    return AI.solve(
        problem, iteration_algorithm; iterate = copy(state), env = copy(env)
    )
end

# === Layer 1: apply_operators iteration ===

@kwdef struct ApplyOperatorsProblem{Ops, Init} <: AI.Problem
    operators::Ops
    init::Init
end

@kwdef struct ApplyOperatorsAlgorithm{
        OperatorAlgorithm,
        EnvironmentAlgorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AI.Algorithm
    operator_algorithm::OperatorAlgorithm
    environment_algorithm::EnvironmentAlgorithm = NoEnvironmentPreparation()
    # Placeholder default; the operator-count bound is filled in per call by
    # `apply_operators` (where `length(operators)` is known).
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(0)
end

@kwdef mutable struct ApplyOperatorsState{
        Iterate, Env, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    env::Env
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm;
        iterate, env, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return ApplyOperatorsState(;
        iterate, env, iteration, stopping_criterion_state
    )
end

function AI.initialize_state!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm,
        state::ApplyOperatorsState; iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion,
        state.stopping_criterion_state
    )
    return state
end

function AI.step!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm,
        state::ApplyOperatorsState
    )
    state.iterate, state.env = prepare_environment(
        algorithm.environment_algorithm, algorithm.operator_algorithm,
        problem.operators, state.iteration, state.iterate, state.env
    )
    state.iterate, state.env = apply_operator(
        algorithm.operator_algorithm, problem.operators[state.iteration], state.iterate,
        state.env
    )
    return state
end

function AI.finalize_state!(
        ::ApplyOperatorsProblem, ::ApplyOperatorsAlgorithm, state::ApplyOperatorsState
    )
    return state.iterate, state.env
end

# === Layer 2: environment-preparation strategy ===

# Before each operator is applied, `prepare_environment` brings the environment
# (and possibly the factors) up to date with the current state, so the upcoming
# `apply_operator` sees a consistent gauge. Strategies subtype
# `EnvironmentPreparationAlgorithm` and overload
#
#     prepare_environment(alg, operator_algorithm, operators, iteration, iterate, env)
#         -> (iterate, env)
#
# `operators` and `iteration` give the full gate sequence and the current
# position (so a strategy can look at the previous/upcoming gates to judge which
# messages went stale), and `operator_algorithm` lets it condition on how the
# gate will be applied (e.g. skip reconvergence for an untruncated/unitary gate).
# A strategy may also return updated factors, since regauging/orthogonalizing can
# rewrite the tensors themselves. On a loopy graph the stale region is not
# sharply defined, so the strategy — not a fixed dirty-set on the cache — owns
# the decision of what to recompute.
#
# Only the no-op is implemented for now; reconvergence policies (local BP around
# the gate support, path reconvergence on a tree, full BP) are left to follow-up
# work.
abstract type EnvironmentPreparationAlgorithm <: AbstractAlgorithm end

struct NoEnvironmentPreparation <: EnvironmentPreparationAlgorithm end

function prepare_environment(
        ::NoEnvironmentPreparation, operator_algorithm, operators, iteration, iterate, env
    )
    return iterate, env
end

function default_algorithm(::typeof(prepare_environment), ::Type{<:Tuple}; kwargs...)
    return NoEnvironmentPreparation()
end

# === Layer 3: single-operator strategy ===

abstract type ApplyOperatorAlgorithm <: AbstractAlgorithm end

function apply_operator(operator, state, env; alg = nothing, kwargs...)
    algorithm = select_algorithm(apply_operator, alg, (operator, state, env); kwargs...)
    return apply_operator(algorithm, operator, state, env)
end

# Out-of-place per-operator step: `initialize_output` allocates fresh `iterate`
# and `env` buffers (copies of the inputs) that `apply_operator!` fills in place,
# leaving the inputs untouched. Returns the new `(iterate, env)` pair.
function apply_operator(algorithm::ApplyOperatorAlgorithm, operator, state, env)
    dest, env_dest = initialize_output(apply_operator!, algorithm, operator, state, env)
    apply_operator!(algorithm, dest, operator, state, env_dest)
    return dest, env_dest
end

# === Default strategy: BPApplyGate ===

@kwdef struct BPApplyGate{Trunc, Pinv <: NamedTuple} <: ApplyOperatorAlgorithm
    trunc::Trunc = nothing
    pinv::Pinv = (;)
    normalize::Bool = false
end

function apply_operator!(
        algorithm::BPApplyGate, dest, operator, state, env
    )
    apply_gate_bp!(
        dest, operator, state, env;
        algorithm.trunc, algorithm.pinv, algorithm.normalize
    )
    return dest
end

function initialize_output(
        ::typeof(apply_operator!), ::BPApplyGate, operator, state, env
    )
    return copy(state), copy(env)
end

function default_algorithm(::typeof(apply_operator), ::Type{<:Tuple}; kwargs...)
    return BPApplyGate(; kwargs...)
end

# === BP simple-update implementation ===

function apply_gate_bp!(
        dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, env; kwargs...
    )
    op_in = domainnames(op)
    vs = [v for v in vertices(state) if !isempty(intersect(op_in, sitenames(state, v)))]
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    return apply_gate_bp_nsite!(Val(length(vs)), dest, op, state, env, vs; kwargs...)
end

function apply_gate_bp_nsite!(
        ::Val{N}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, env, vs; kwargs...
    ) where {N}
    return throw(ArgumentError("$N-site gate decomposition not implemented"))
end

function apply_gate_bp_nsite!(
        ::Val{1}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, env, vs;
        normalize, kwargs...
    )
    v = only(vs)
    ψv = NDA.apply(op, state[v])
    if normalize
        gauges = [
            gram_eigh_full(env[e])
                for e in boundary_edges(state, vs; dir = :in)
        ]
        ψv /= norm(prod([[ψv]; gauges]))
    end
    dest[v] = ψv
    return dest
end

function apply_gate_bp_nsite!(
        ::Val{2}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, env, vs;
        trunc, pinv, normalize
    )
    v1, v2 = vs
    edges_in = boundary_edges(state, vs; dir = :in)
    grams_v1 =
        [gram_eigh_full_with_pinv(env[e]; pinv) for e in edges_in if dst(e) == v1]
    grams_v2 =
        [gram_eigh_full_with_pinv(env[e]; pinv) for e in edges_in if dst(e) == v2]
    gauges_v1, inv_gauges_v1 = first.(grams_v1), last.(grams_v1)
    gauges_v2, inv_gauges_v2 = first.(grams_v2), last.(grams_v2)

    ψ_v1 = prod([[state[v1]]; gauges_v1])
    ψ_v2 = prod([[state[v2]]; gauges_v2])

    Q_v1, R_v1 = TA.qr(ψ_v1, setdiff(dimnames(ψ_v1), dimnames(ψ_v2), dimnames(op)))
    Q_v2, R_v2 = TA.qr(ψ_v2, setdiff(dimnames(ψ_v2), dimnames(ψ_v1), dimnames(op)))
    op_R_v1v2 = NDA.apply(op, R_v1 * R_v2)
    U_v1, S, U_v2 = TA.svd(op_R_v1v2, setdiff(dimnames(R_v1), dimnames(R_v2)); trunc)
    if normalize
        S = S / norm(S)
    end
    name_v1, name_v2 = dimnames(S)
    sqrt_S = sqrt(S, (name_v1,), (name_v2,))
    R_v1 = replacedimnames(U_v1 * sqrt_S, name_v2 => name_v1)
    R_v2 = sqrt_S * U_v2

    dest[v1] = prod([[Q_v1 * R_v1]; inv_gauges_v1])
    dest[v2] = prod([[Q_v2 * R_v2]; inv_gauges_v2])

    fresh_12 = randname(name_v1)
    fresh_21 = randname(name_v1)
    env[v1 => v2] =
        operator(replacedimnames(S, name_v2 => fresh_12), (name_v1,), (fresh_12,))
    env[v2 => v1] =
        operator(replacedimnames(S, name_v2 => fresh_21), (name_v1,), (fresh_21,))
    return dest
end
