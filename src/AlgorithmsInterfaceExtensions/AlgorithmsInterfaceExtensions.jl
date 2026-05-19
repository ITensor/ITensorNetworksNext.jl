module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI

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
