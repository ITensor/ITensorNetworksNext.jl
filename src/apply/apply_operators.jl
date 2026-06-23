using .AlgorithmsInterfaceExtensions: AlgorithmsInterfaceExtensions as AIE
using AlgorithmsInterface: AlgorithmsInterface as AI
using Base: @kwdef
using Graphs: dst, src, vertices
using ITensorBase:
    ITensorBase as ITB, AbstractITensor, dimnames, domainnames, operator, replacedimnames
using LinearAlgebra: norm
using NamedGraphs.GraphsExtensions: all_edges, boundary_edges
using TensorAlgebra: TensorAlgebra as TA, gram_eigh_full, gram_eigh_full_with_pinv

# === Top-level user entry point ===

"""
    apply_operators(operators, state, env; alg=nothing, kwargs...) -> (state, env)

Apply each operator in `operators` (a sequence of single-tensor or two-tensor
operators) to `state` in turn, updating `env` to reflect each application.
`state` is an `AbstractITensorNetwork`, `env` is a per-edge environment cache
(typically built by `identity_norm_message_env(state)` or one of the related
`*_norm_message_env` constructors), and the returned `(state, env)` pair has
the operators applied. `kwargs` are forwarded to the per-operator algorithm
(`alg`); for the default BP simple-update algorithm these include `trunc`
(forwarded to the SVD that splits a two-site gate back into single-site
tensors) and `normalize`.

See also [`apply_operator`](@ref).
"""
function apply_operators(operators, state, env; alg = nothing, kwargs...)
    algorithm = select_algorithm(
        apply_operators, alg, (operators, state, env); kwargs...
    )
    return apply_operators(algorithm, operators, state, env)
end

# The `apply_operators` iteration algorithm wraps the per-operator algorithm,
# which is itself resolved via `apply_operator` (overridable with `operator_alg`).
function default_algorithm(
        ::typeof(apply_operators), args::Tuple;
        operator_alg = nothing, environment_alg = nothing, kwargs...
    )
    operators, state, env = args
    # `apply_operator` acts on a single operator, so select on the operator
    # element type, keeping the remaining `(state, env)` argument types.
    # We use types here in case the operator list is empty.
    operator_args = Tuple{eltype(operators), typeof(state), typeof(env)}
    operator_algorithm =
        select_algorithm(apply_operator, operator_alg, operator_args; kwargs...)
    # `apply_operator_environment_preparation` signature (minus the env algorithm):
    # `(operator_algorithm, operators, iteration::Int, iterate, env)`.
    prepare_args = (operator_algorithm, operators, 0, state, env)
    environment_algorithm = select_algorithm(
        apply_operator_environment_preparation, environment_alg, prepare_args
    )
    return ApplyOperatorsAlgorithm(;
        operator_algorithm,
        environment_algorithm,
        stopping_criterion = AI.StopAfterIteration(length(operators))
    )
end

function apply_operators(algorithm, operators, state, env)
    isempty(operators) && return copy(state), copy(env)
    problem = ApplyOperatorsProblem(; operators, init = state)
    return AI.solve(problem, algorithm; iterate = state, env)
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
    environment_algorithm::EnvironmentAlgorithm = NoApplyOperatorEnvironmentPreparation()
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

function AI.step!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm,
        state::ApplyOperatorsState
    )
    # Prepare for the operator application, for example by updating the
    # environments in a path between where the operators are being applied.
    state.iterate, state.env = apply_operator_environment_preparation(
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

# Update the environment (and possibly the factors) before the next operator is
# applied. The full `operators`/`iteration` and `operator_algorithm` are passed so
# a strategy can judge which messages went stale and how much to recompute; it may
# also return regauged/orthogonalized factors. Only the no-op is implemented for
# now (reconvergence policies are follow-up work).
struct NoApplyOperatorEnvironmentPreparation <: AbstractAlgorithm end

function apply_operator_environment_preparation(
        ::NoApplyOperatorEnvironmentPreparation, operator_algorithm, operators, iteration,
        iterate, env
    )
    return iterate, env
end

function default_algorithm(
        ::typeof(apply_operator_environment_preparation), ::Type{<:Tuple}; kwargs...
    )
    return NoApplyOperatorEnvironmentPreparation()
end

# === Layer 3: single-operator strategy ===

abstract type ApplyOperatorAlgorithm <: AbstractAlgorithm end

"""
    apply_operator(operator, state, env; alg=nothing, kwargs...) -> (state, env)

Apply a single `operator` to `state` and return an updated `(state, env)` pair.
Environments are not fully recomputed; only the edges touched by `operator`
are updated (for a two-site gate, the BP simple-update default writes new
messages on the gate edge). For the BP simple-update default algorithm,
`kwargs` accept `trunc` (forwarded to the SVD that splits the gate back into
single-site tensors) and `normalize` (whether to rescale the post-gate state
so the singular-value spectrum stays unit-norm).

See also [`apply_operators`](@ref).
"""
function apply_operator(operator, state, env; alg = nothing, kwargs...)
    algorithm = select_algorithm(apply_operator, alg, (operator, state, env); kwargs...)
    return apply_operator(algorithm, operator, state, env)
end

function apply_operator(algorithm::ApplyOperatorAlgorithm, operator, state, env)
    dest, env_dest = initialize_output(apply_operator!, algorithm, operator, state, env)
    apply_operator!(algorithm, dest, operator, state, env_dest)
    return dest, env_dest
end

# === Default strategy: BPApplyGate ===

@kwdef struct BPApplyGate{Trunc} <: ApplyOperatorAlgorithm
    trunc::Trunc = nothing
    normalize::Bool = false
end

function apply_operator!(
        algorithm::BPApplyGate, dest, operator, state, env
    )
    apply_gate_bp!(
        dest, operator, state, env;
        algorithm.trunc, algorithm.normalize
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
        dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env; kwargs...
    )
    op_in = domainnames(op)
    vs = [v for v in vertices(state) if !isempty(intersect(op_in, sitenames(state, v)))]
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    return apply_gate_bp_nsite!(Val(length(vs)), dest, op, state, env, vs; kwargs...)
end

function apply_gate_bp_nsite!(
        ::Val{N}, dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env, vs; kwargs...
    ) where {N}
    return throw(ArgumentError("$N-site gate decomposition not implemented"))
end

function apply_gate_bp_nsite!(
        ::Val{1}, dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env, vs;
        normalize, kwargs...
    )
    v = only(vs)
    ψv = ITB.apply(op, state[v])
    if normalize
        gauges = [
            conj(gram_eigh_full(env[e]))
                for e in boundary_edges(state, vs; dir = :in)
        ]
        ψv /= norm(prod([[ψv]; gauges]))
    end
    dest[v] = ψv
    return dest
end

function apply_gate_bp_nsite!(
        ::Val{2}, dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env, vs;
        trunc, normalize
    )
    v1, v2 = vs
    edges_in = boundary_edges(state, vs; dir = :in)
    grams_v1 =
        [gram_eigh_full_with_pinv(env[e]) for e in edges_in if dst(e) == v1]
    grams_v2 =
        [gram_eigh_full_with_pinv(env[e]) for e in edges_in if dst(e) == v2]
    gauges_v1, inv_gauges_v1 = conj.(first.(grams_v1)), conj.(last.(grams_v1))
    gauges_v2, inv_gauges_v2 = conj.(first.(grams_v2)), conj.(last.(grams_v2))

    ψ_v1 = prod([[state[v1]]; gauges_v1])
    ψ_v2 = prod([[state[v2]]; gauges_v2])

    Q_v1, R_v1 = TA.qr(ψ_v1, setdiff(dimnames(ψ_v1), dimnames(ψ_v2), dimnames(op)))
    Q_v2, R_v2 = TA.qr(ψ_v2, setdiff(dimnames(ψ_v2), dimnames(ψ_v1), dimnames(op)))
    op_R_v1v2 = ITB.apply(op, R_v1 * R_v2)
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

    env[v1 => v2] = operator(conj(S), (name_v2,), (name_v1,))
    env[v2 => v1] = operator(
        conj(replacedimnames(S, name_v1 => name_v2, name_v2 => name_v1)),
        (name_v2,), (name_v1,)
    )
    return dest
end
