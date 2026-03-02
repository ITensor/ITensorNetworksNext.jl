module ITensorNetworksNextDaggerExt

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import ITensorNetworksNext.ITensorNetworksNextParallel as ITNNP
using Dagger
using ITensorNetworksNext.ITensorNetworksNextParallel:
    DaggerNestedAlgorithm, DaggerState, ITensorNetworksNextParallel
using Dictionaries: set!

function ITNNP.DaggerNestedAlgorithm(f, iterable; kwargs...)
    return DaggerNestedAlgorithm(; algorithms = map(f, iterable), kwargs...)
end

function ITNNP.dagger_algorithm(f::Base.Callable, iterable; kwargs...)
    return DaggerNestedAlgorithm(f, iterable; kwargs...)
end

function ITNNP.initialize_dagger_state(
        problem::AIE.Problem, algorithm::AIE.Algorithm; iterate
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )

    remote_results = Dictionary{Int, Dagger.DTask}()

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
    return ITNNP.initialize_dagger_state(problem, algorithm; kwargs...)
end

function AI.step!(
        problem::AIE.Problem,
        algorithm::ITNNP.DaggerNestedAlgorithm,
        state::ITNNP.DaggerState;
        kwargs...
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]

    iterate = ITNNP.get_subiterate(subproblem, subalgorithm, state)

    dtask = Dagger.@spawn AI.solve(subproblem, subalgorithm; iterate)

    set!(state.remote_results, state.iteration, dtask)

    return state
end

include("daggerbeliefpropagation.jl")

end # ITensorNetworksNextDaggerExt
