import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
import MatrixAlgebraKit as MAK
import NamedDimsArrays as NDA
import TensorAlgebra as TA
using Base: @kwdef
using Graphs: dst, src, vertices
using LinearAlgebra: norm
using NamedDimsArrays:
    AbstractNamedDimsArray, dimnames, domainnames, nameddims, randname, replacedimnames
using NamedGraphs.GraphsExtensions: all_edges, boundary_edges

# === Top-level user entry point ===

function apply_operators(operators, state; op_alg = nothing, kwargs...)
    op_alg = AIE.select_algorithm(apply_operator!, op_alg, (state,))
    problem = ApplyOperatorsProblem(; operators, init = state)
    algorithm = ApplyOperatorsAlgorithm(;
        operator_algorithm = op_alg,
        stopping_criterion = AI.StopAfterIteration(length(operators))
    )
    return AI.solve(problem, algorithm; iterate = copy(state), kwargs...)
end

function apply_operator(operator, state; alg = nothing, kwargs...)
    return apply_operators([operator], state; op_alg = alg, kwargs...)
end

# === Layer 1: apply_operators iteration ===

@kwdef struct ApplyOperatorsProblem{Ops, Init} <: AI.Problem
    operators::Ops
    init::Init
end

@kwdef struct ApplyOperatorsAlgorithm{
        OperatorAlgorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AI.Algorithm
    operator_algorithm::OperatorAlgorithm
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct ApplyOperatorsState{
        Iterate, EnvCache, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    env_cache::EnvCache
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm;
        iterate, env_cache!, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return ApplyOperatorsState(;
        iterate, env_cache = env_cache!, iteration, stopping_criterion_state
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
    op = problem.operators[state.iteration]
    apply_operator!(
        algorithm.operator_algorithm, state.iterate, op, state.iterate;
        env_cache! = state.env_cache
    )
    return state
end

# === Layer 2: single-operator strategy ===

abstract type ApplyOperatorAlgorithm <: AIE.AbstractAlgorithm end

function apply_operator! end

function AIE.default_algorithm(::typeof(apply_operator!), ::Type{<:Tuple}; kwargs...)
    return BPApplyGate(; kwargs...)
end

function apply_operator(algorithm::ApplyOperatorAlgorithm, operator, state; env_cache!)
    dest = AIE.initialize_output(apply_operator!, algorithm, operator, state)
    return apply_operator!(algorithm, dest, operator, state; env_cache!)
end

# === Default strategy: BPApplyGate ===

@kwdef struct BPApplyGate{Trunc, PinvKwargs <: NamedTuple} <: ApplyOperatorAlgorithm
    trunc::Trunc = nothing
    pinv_kwargs::PinvKwargs = (; tol = 0)
    normalize::Bool = false
end

function AIE.initialize_output(
        ::typeof(apply_operator!), ::BPApplyGate, operator, state
    )
    return copy(state)
end

function apply_operator!(
        algorithm::BPApplyGate, dest, operator, state; env_cache!
    )
    apply_gate_bp!(
        dest, operator, state;
        sqrt_messages! = env_cache!,
        algorithm.trunc, algorithm.pinv_kwargs, algorithm.normalize
    )
    return dest
end

# === BP simple-update implementation ===

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
        sqrt_messages!, normalize, kwargs...
    )
    v = only(vs)
    ψv = NDA.apply(op, state[v])
    if normalize
        sqrt_envs =
            [sqrt_messages![e] for e in boundary_edges(sqrt_messages!, vs; dir = :in)]
        ψv /= norm(prod([[ψv]; sqrt_envs]))
    end
    dest[v] = ψv
    return dest
end

function apply_gate_bp_nsite!(
        ::Val{2}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs;
        sqrt_messages!, trunc, pinv_kwargs, normalize
    )
    v1, v2 = vs
    edges_in = boundary_edges(sqrt_messages!, vs; dir = :in)
    sqrt_envs_v1 = [sqrt_messages![e] for e in edges_in if dst(e) == v1]
    sqrt_envs_v2 = [sqrt_messages![e] for e in edges_in if dst(e) == v2]

    ψ_v1 = prod([[state[v1]]; sqrt_envs_v1])
    ψ_v2 = prod([[state[v2]]; sqrt_envs_v2])

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
    dest[v1] = prod([[Q_v1 * R_v1]; inv_sqrt_envs_v1])
    dest[v2] = prod([[Q_v2 * R_v2]; inv_sqrt_envs_v2])

    sqrt_messages![v1 => v2] = replacedimnames(
        sqrt_S, name_v1 => randname(name_v1), name_v2 => name_v1
    )
    sqrt_messages![v2 => v1] = replacedimnames(sqrt_S, name_v2 => randname(name_v2))
    return dest
end
