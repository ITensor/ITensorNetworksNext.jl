module ITensorNetworksNextDaggerExt

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import ITensorNetworksNext.ITensorNetworksNextParallel as ITNNP
using Dagger
using Dictionaries: set!
using ITensorNetworksNext.ITensorNetworksNextParallel:
    DaggerNestedAlgorithm, DaggerState, ITensorNetworksNextParallel, step_dagger!

function AI.initialize_state(problem::AI.Problem, algorithm::AI.Algorithm; kwargs...)
    substate = AI.initialize_state(problem, algorithm; kwargs...)
    remote_results = Dictionary{Int, Dagger.DTask}()
    return DaggerState(;
        substate,
        substate.iteration,
        substate.stopping_criterion_state,
        remote_results
    )
end

function AI.step!(
        problem::AI.Problem,
        algorithm::Dagger,
        state::DaggerState
    ) where {Algorithm}
    # Forward the "external" stopping info to the internal states.
    state.substate.iteration = state.iteration
    state.substate.stopping_criterion_state = state.stopping_criterion_state

    dtask = Dagger.@spawn step_dagger!(problem, algorithm.parent, substate)

    set!(state.remote_results, state.iteration, dtask)
    return state
end

function ITNNP.step_dagger!(
        problem::AI.Problem,
        algorithm::AI.Algorithm,
        state::AI.State
    )
    AI.step!(problem, algorithm, state)

    return return state
end

function AI.finalize_state!(::AI.Problem, ::AI.Algorithm, state::DaggerState)
    for dtask in collect(state.remote_results)
        wait(dtask)
    end
    return state.substate.iterate
end

# include("daggerbeliefpropagation.jl")

end # ITensorNetworksNextDaggerExt
