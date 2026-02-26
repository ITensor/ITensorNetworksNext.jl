module ITensorNetworksNextDaggerExt

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import ITensorNetworksNext.ITensorNetworksNextParallel as ITNNP
using Dagger
using ITensorNetworksNext.ITensorNetworksNextParallel:
    DaggerNestedAlgorithm, DaggerState, ITensorNetworksNextParallel

function ITNNP.DaggerNestedAlgorithm(f::Function, iterable; workers = workers(), kwargs...)
    return DaggerNestedAlgorithm(; algorithms = map(f, iterable), workers, kwargs...)
end

function initialize_dagger_state(problem::AIE.Problem, algorithm::AIE.Algorithm; iterate)
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )

    remote_results = Dict{Int, Dagger.DTask}()

    return ITNNP.DaggerState(;
        iterate,
        remote_results,
        stopping_criterion_state
    )
end

function AI.initialize_state(
        problem::AIE.Problem,
        algorithm::ITNNP.DaggerNestedAlgorithm;
        kwargs...
    )
    return initialize_dagger_state(problem, algorithm; kwargs...)
end

function AIE.get_subproblem(
        problem::AIE.Problem,
        algorithm::ITNNP.DaggerNestedAlgorithm,
        state::ITNNP.DaggerState
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]

    # This might be a Dagger.chun object.
    iterate = ITNNP.get_subiterate(subproblem, subalgorithm, state)

    substate = Dagger.@spawn AI.initialize_state(subproblem, subalgorithm; iterate)

    return subproblem, subalgorithm, substate
end

function AI.step!(
        problem::AI.Problem,
        algorithm::ITNNP.DaggerNestedAlgorithm,
        state::ITNNP.DaggerState;
        kwargs...
    )
    subproblem, subalgorithm, substate_future =
        AIE.get_subproblem(problem, algorithm, state)

    dtask = Dagger.@spawn AI.solve(subproblem, subalgorithm, substate_future)

    state.remote_results[state.iteration] = dtask

    return state
end

include("daggerbeliefpropagation.jl")

end # ITensorNetworksNextDaggerExt
