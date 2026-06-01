module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI

@kwdef mutable struct GenericAlgorithmState{Iterate, StoppingCriterionState} <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::AI.Problem,
        algorithm::AI.Algorithm;
        iterate,
        iteration = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return GenericAlgorithmState(; iterate, iteration, stopping_criterion_state)
end

function AI.initialize_state!(
        problem::AI.Problem,
        algorithm::AI.Algorithm,
        state::AI.State;
        iteration = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

# ============================ NestedAlgorithm =============================================

abstract type NestedAlgorithm <: AI.Algorithm end

# Subtypes of `NestedAlgorithm` must override `initialize_subsolve` — it
# returns the `(subproblem, subalgorithm, substate)` tuple that the next
# inner `AI.solve!` call consumes. The default `finalize_substate!` copies
# the substate's iterate back into the parent state; subtypes can override
# when more is required.
function initialize_subsolve(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State
    )
    return throw(MethodError(initialize_subsolve, (problem, algorithm, state)))
end

function finalize_substate!(
        ::AI.Problem, ::AI.Algorithm, substate::AI.State, state::AI.State
    )
    state.iterate = substate.iterate
    return state
end

function AI.step!(problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State)
    subproblem, subalgorithm, substate = initialize_subsolve(problem, algorithm, state)
    AI.solve!(subproblem, subalgorithm, substate)
    finalize_substate!(subproblem, subalgorithm, substate, state)
    return state
end

# ============================ NestedState =================================================

# State that wraps an inner `substate` and forwards `:iterate` accesses to it,
# so the inner-loop iterate is shared without duplicating storage on the outer
# state. Subtypes must store the inner state as a field named `substate`.
abstract type NestedState <: AI.State end

@kwdef mutable struct GenericNestedState{Substate, StoppingCriterionState} <: NestedState
    substate::Substate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

# Use `getfield` on the right-hand side so future edits to this forwarder
# can't accidentally recurse through the overload.
function Base.getproperty(state::NestedState, name::Symbol)
    name === :iterate && return getfield(state, :substate).iterate
    return getfield(state, name)
end
function Base.setproperty!(state::NestedState, name::Symbol, value)
    name === :iterate && return (getfield(state, :substate).iterate = value)
    return setfield!(state, name, value)
end
function Base.propertynames(state::NestedState)
    return (fieldnames(typeof(state))..., :iterate)
end

# ============================ select_algorithm / default_algorithm ========================

# Like `MatrixAlgebraKit.select_algorithm` / `default_algorithm`, but
# selection-relevant inputs are packed into an `args` tuple so the value
# and type domains stay disjoint: `(1.2,)` vs `Tuple{Float64}`. Strategy
# types subtype `AbstractAlgorithm` so the passthrough overload is generic.
abstract type AbstractAlgorithm end

function default_algorithm(f, ::Type{Args}; kwargs...) where {Args <: Tuple}
    return throw(MethodError(default_algorithm, (f, Args)))
end
function default_algorithm(f, args::Tuple; kwargs...)
    return default_algorithm(f, typeof(args); kwargs...)
end

function select_algorithm(f, alg, args::Tuple; kwargs...)
    return select_algorithm(f, alg, typeof(args); kwargs...)
end
function select_algorithm(f, ::Nothing, ::Type{Args}; kwargs...) where {Args <: Tuple}
    return default_algorithm(f, Args; kwargs...)
end
function select_algorithm(f, alg::NamedTuple, ::Type{Args}; kwargs...) where {Args <: Tuple}
    isempty(kwargs) || throw(
        ArgumentError(
            "Additional keyword arguments are not allowed when `alg` is a `NamedTuple`."
        )
    )
    return default_algorithm(f, Args; alg...)
end
function select_algorithm(f, alg::AbstractAlgorithm, ::Type{<:Tuple}; kwargs...)
    isempty(kwargs) || throw(
        ArgumentError(
            "Additional keyword arguments are not allowed when `alg` is an `AbstractAlgorithm` instance."
        )
    )
    return alg
end

# ============================ StopWhenConverged ===========================================

# Stopping criterion that fires once `iterate_diff(iterate, previous_iterate) < tol`.
# Concrete iterate types must supply an `iterate_diff` method.
function iterate_diff(a, b)
    return throw(MethodError(iterate_diff, (a, b)))
end

@kwdef struct StopWhenConverged <: AI.StoppingCriterion
    tol::Float64
end

@kwdef mutable struct StopWhenConvergedState{Iterate} <: AI.StoppingCriterionState
    delta::Float64 = Inf
    at_iteration::Int = -1
    previous_iterate::Iterate
end

function AI.initialize_state(::AI.Problem, ::AI.Algorithm, ::StopWhenConverged; iterate)
    return StopWhenConvergedState(; previous_iterate = copy(iterate))
end

function AI.initialize_state!(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenConverged, st::StopWhenConvergedState
    )
    st.delta = Inf
    return st
end

function AI.is_finished!(
        problem::AI.Problem,
        algorithm::AI.Algorithm,
        state::AI.State,
        c::StopWhenConverged,
        st::StopWhenConvergedState
    )
    iterate = state.iterate
    previous_iterate = st.previous_iterate

    delta = iterate_diff(iterate, previous_iterate)

    st.previous_iterate = copy(iterate)

    # delta = 0 initially, so skip this the first time.
    state.iteration == 0 && return false

    st.delta = delta

    if AI.is_finished(problem, algorithm, state, c, st)
        st.at_iteration = state.iteration
        return true
    end

    return false
end

function AI.is_finished(
        ::AI.Problem,
        ::AI.Algorithm,
        ::AI.State,
        c::StopWhenConverged,
        st::StopWhenConvergedState
    )
    return st.delta < c.tol
end

end
