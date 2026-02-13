module ITensorNetworksNextDaggerExt

using Dagger
using Dagger.Distributed
using ITensorNetworksNext.ITensorNetworksNextParallel: DaggerNestedAlgorithm, DaggerState,
    ITensorNetworksNextParallel

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE


function ITensorNetworksNextParallel.DaggerNestedAlgorithm(f::Function, iterable; workers = workers(), kwargs...)
    return DaggerNestedAlgorithm(; algorithms = map(f, iterable), workers, kwargs...)
end

function initialize_dagger_state(
        problem::AIE.Problem,
        algorithm::AIE.Algorithm;
        iterate,
        remote_subiterates = Dict{Int, Dagger.Chunk}(),
    )

    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )

    remote_results = Dict{Int, Dagger.DTask}()

    return DaggerState(; iterate, remote_subiterates, stopping_criterion_state, remote_results)
end

function AI.initialize_state(
        problem::AIE.Problem,
        algorithm::DaggerNestedAlgorithm;
        kwargs...
    )
    return initialize_dagger_state(problem, algorithm; kwargs...)
end

function AIE.get_subproblem(
        problem::AIE.Problem,
        algorithm::AIE.NestedAlgorithm,
        state::DaggerState
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]

    iterate = state.iterate
    remote_subiterates = state.remote_subiterates

    substate = AI.initialize_state(subproblem, subalgorithm; iterate, remote_subiterates)

    return subproblem, subalgorithm, substate
end


function AI.step!(
        problem::AI.Problem,
        algorithm::DaggerNestedAlgorithm,
        state::DaggerState;
        kwargs...
    )

    subproblem, subalgorithm, subiterate_chunk = AIE.get_subproblem(problem, algorithm, state)

    dtask = Dagger.@spawn AI.solve(subproblem, subalgorithm; iterate = subiterate_chunk)

    AIE.set_substate!(problem, algorithm, state, dtask)

    return state
end

function AIE.set_substate!(
        ::AIE.Problem,
        ::DaggerNestedAlgorithm,
        state::DaggerState,
        dtask::Dagger.DTask,
    )
    state.remote_results[state.iteration] = dtask

    return state
end

include("daggerbeliefpropagation.jl")

end # ITensorNetworksNextDaggerExt
