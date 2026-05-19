module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI

# ========================== Patches for AlgorithmsInterface.jl ============================

abstract type Problem <: AI.Problem end
abstract type Algorithm <: AI.Algorithm end
abstract type State <: AI.State end

function AI.initialize_state!(
        problem::Problem, algorithm::Algorithm, state::State; iteration = 0, kwargs...
    )
    for (k, v) in pairs(kwargs)
        setproperty!(state, k, v)
    end
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AI.initialize_state(
        problem::Problem, algorithm::Algorithm; iterate, kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return DefaultState(; iterate, stopping_criterion_state, kwargs...)
end

# ============================ DefaultState ================================================

@kwdef mutable struct DefaultState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

# ============================ increment! ==================================================

# Custom version of `increment!` that also takes the problem and algorithm as arguments.
function AI.increment!(problem::Problem, algorithm::Algorithm, state::State)
    return AI.increment!(state)
end

# ============================ NestedAlgorithm =============================================

abstract type NestedAlgorithm <: Algorithm end

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

end
