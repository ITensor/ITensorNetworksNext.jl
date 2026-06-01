module ITensorNetworksNextDaggerExt

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import ITensorNetworksNext.ITensorNetworksNextParallel as ITNNP
using Dagger
using Dictionaries: Dictionary, set!

@kwdef mutable struct DaggerState{Chunk <: Dagger.Chunk} <: AI.State
    chunk::Chunk # A chunk living on the host worker.
    futures = Dictionary{Int, Dagger.DTask}() # the futures from the remote steps.
end

function Base.getproperty(state::DaggerState, name::Symbol)
    if name in (:chunk, :futures)
        return getfield(state, name)
    end
    return getproperty(fetch(state.chunk), name)
end
function Base.setproperty!(state::DaggerState, name::Symbol, val)
    if name === (:chunk, :futures)
        return setfield!(state, name, val)
    end
    fetch(Dagger.@spawn setproperty!(state.chunk, name, val))
    return state
end

# ====================================== overloads ======================================= #

function ITNNP.default_workers(::AI.Algorithm, ::ITNNP.AbstractDaggerStrategy)
    return Dagger.Distributed.workers()
end

function ITNNP.initialize_parallel_state(
        problem::AI.Problem, algorithm::AI.Algorithm,
        _strategy::ITNNP.GenericDaggerStrategy; kwargs...
    )
    chunk = Dagger.@mutable AI.initialize_state(problem, algorithm; iterate, kwargs...)

    return DaggerState(; chunk)
end

function AI.step!(problem::AI.Problem, algorithm::ITNNP.Parallelized, state::DaggerState)
    worker_list = algorithm.workers
    algorithm = algorithm.parent

    function remote_solve(state)
        subsolve = AIE.initialize_subsolve(problem, algorithm, state)

        subproblem, subalgorithm, substate = subsolve
        AI.solve!(subproblem, subalgorithm, substate)

        return subsolve
    end

    worker = worker_list[mod(state.iteration, 1:length(worker_list))]

    dtask = Dagger.@spawn scope = Dagger.scope(; worker) remote_solve(fetch(state.chunk))

    # Spawns on the host only (as we do not fetch the chunk before hand.)
    dtask = Dagger.spawn(dtask, state.chunk) do subsolve, state
        subproblem, subalgorithm, substate = subsolve

        AIE.finalize_substate!(subproblem, subalgorithm, substate, state)
        return state
    end

    set!(state.futures, state.iteration, dtask)

    return state
end
function AI.finalize_state!(::AI.Problem, ::AI.Algorithm, state::DaggerState)
    foreach(fetch, state.futures)
    return state.iterate
end

function AIE.finalize_substate!(
        problem::AI.Problem, algorithm::AI.Algorithm, substate::DaggerState, state::AI.State
    )
    AIE.finalize_substate!(problem, algorithm, fetch(substate.chunk), state)
    return state
end

end # ITensorNetworksNextDaggerExt
