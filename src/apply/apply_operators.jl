import AlgorithmsInterface as AI
import NamedDimsArrays as NDA
using Base: @kwdef
using Graphs: vertices
using LinearAlgebra: norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, domainnames
using NamedGraphs.GraphsExtensions: boundary_edges
using TensorAlgebra: TensorAlgebra

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
    return throw(
        MethodError(finalize_substate!, (problem, algorithm, state, substate))
    )
end

function AI.step!(problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State)
    subproblem, subalgorithm, substate = initialize_subproblem(problem, algorithm, state)
    AI.solve!(subproblem, subalgorithm, substate)
    finalize_substate!(problem, algorithm, state, substate)
    return state
end

# === apply_operators (plural, iterative over a list of operators) ===

function apply_operators(ops, state; op_alg = BPApplyOperator(), kwargs...)
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
        subproblem, subalgorithm; iterate = state.iterate, cache! = state.cache
    )
    return subproblem, subalgorithm, substate
end

function finalize_substate!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperators,
        state::ApplyOperatorsState, substate::AI.State
    )
    state.iterate = substate.iterate
    return state
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

function apply_operator(op, state; alg = BPApplyOperator(), kwargs...)
    problem = ApplyOperatorProblem(; op, init = state)
    return AI.solve(problem, alg; iterate = copy(state), kwargs...)
end

function apply_operator!(dest, op, state; alg = BPApplyOperator(), kwargs...)
    problem = ApplyOperatorProblem(; op, init = state)
    alg_state = AI.initialize_state(problem, alg; iterate = dest, kwargs...)
    return AI.solve!(problem, alg, alg_state)
end

# === BPApplyOperator (non-iterative; overloads solve_loop! directly) ===

@kwdef struct BPApplyOperator{Trunc, PinvKwargs <: NamedTuple} <: AI.Algorithm
    trunc::Trunc = nothing
    pinv_kwargs::PinvKwargs = (; tol = 0)
    normalize::Bool = false
end

@kwdef mutable struct BPApplyOperatorState{Iterate, Cache} <: AI.State
    iterate::Iterate
    cache::Cache
end

function AI.initialize_state(
        problem::ApplyOperatorProblem, algorithm::BPApplyOperator;
        iterate, cache! = initialize_cache(problem, algorithm, iterate)
    )
    return BPApplyOperatorState(; iterate, cache = cache!)
end

# Non-iterative algorithm: no per-call state to reset.
function AI.initialize_state!(
        ::ApplyOperatorProblem, ::BPApplyOperator, state::BPApplyOperatorState
    )
    return state
end

# Non-iterative algorithm: bypass the step!/stopping-criterion loop.
function AI.solve_loop!(
        problem::ApplyOperatorProblem, algorithm::BPApplyOperator,
        state::BPApplyOperatorState
    )
    apply_operator_bp!(
        state.iterate, problem.op, problem.init;
        cache! = state.cache,
        trunc = algorithm.trunc, pinv_kwargs = algorithm.pinv_kwargs,
        normalize = algorithm.normalize
    )
    return state
end

# === BP simple-update implementation ===

function apply_operator_bp!(
        dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork; kwargs...
    )
    op_in = domainnames(op)
    vs = [v for v in vertices(state) if !isempty(intersect(op_in, sitenames(state, v)))]
    isempty(vs) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )
    return apply_operator_bp_nsite!(Val(length(vs)), dest, op, state, vs; kwargs...)
end

function apply_operator_bp_nsite!(
        ::Val{N}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs; kwargs...
    ) where {N}
    throw(ArgumentError("$N-site gate decomposition not implemented"))
end

function apply_operator_bp_nsite!(
        ::Val{1}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs;
        cache!, pinv_kwargs, normalize, kwargs...
    )
    v = only(vs)
    ψv = NDA.apply(op, state[v])
    if normalize
        envs = [cache![e] for e in boundary_edges(cache!, vs; dir = :in)]
        envs_v = filter(e -> !isempty(intersect(dimnames(e), dimnames(state[v]))), envs)
        sqrt_envs_and_invs = map(envs_v) do env
            shared = intersect(dimnames(env), dimnames(state[v]))
            return balanced_eigh_and_inv(
                env, Tuple(setdiff(dimnames(env), shared)), Tuple(shared);
                pinv_kwargs...
            )
        end
        sqrt_envs, inv_sqrt_envs = first.(sqrt_envs_and_invs), last.(sqrt_envs_and_invs)
        ψ_gauge = prod([ψv; sqrt_envs])
        ψv = prod([ψ_gauge / norm(ψ_gauge); inv_sqrt_envs])
    end
    dest[v] = ψv
    return dest
end

function apply_operator_bp_nsite!(
        ::Val{2}, dest::AbstractTensorNetwork, op::AbstractNamedDimsArray,
        state::AbstractTensorNetwork, vs;
        cache!, trunc, pinv_kwargs, normalize
    )
    v1, v2 = vs
    envs = [cache![e] for e in boundary_edges(cache!, vs; dir = :in)]
    envs_v1 = filter(e -> !isempty(intersect(dimnames(e), dimnames(state[v1]))), envs)
    envs_v2 = filter(e -> !isempty(intersect(dimnames(e), dimnames(state[v2]))), envs)
    sqrt_envs_and_invs_v1 = map(envs_v1) do env
        shared = intersect(dimnames(env), dimnames(state[v1]))
        return balanced_eigh_and_inv(
            env, Tuple(setdiff(dimnames(env), shared)), Tuple(shared); pinv_kwargs...
        )
    end
    sqrt_envs_and_invs_v2 = map(envs_v2) do env
        shared = intersect(dimnames(env), dimnames(state[v2]))
        return balanced_eigh_and_inv(
            env, Tuple(setdiff(dimnames(env), shared)), Tuple(shared); pinv_kwargs...
        )
    end
    sqrt_envs_v1, inv_sqrt_envs_v1 =
        first.(sqrt_envs_and_invs_v1), last.(sqrt_envs_and_invs_v1)
    sqrt_envs_v2, inv_sqrt_envs_v2 =
        first.(sqrt_envs_and_invs_v2), last.(sqrt_envs_and_invs_v2)

    ψ_v1 = prod([state[v1]; sqrt_envs_v1])
    ψ_v2 = prod([state[v2]; sqrt_envs_v2])

    s_v1 = sitenames(state, v1)
    s_v2 = sitenames(state, v2)
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
    dest[v1] = ψ_v1
    dest[v2] = ψ_v2
    return dest
end
