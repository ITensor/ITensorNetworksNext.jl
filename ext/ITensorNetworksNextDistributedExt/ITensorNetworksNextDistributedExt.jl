module ITensorNetworksNextDistributedExt

using Distributed

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE

import ITensorNetworksNext.ITensorNetworksNextParallel as Parallel

function initialize_distributed_state(
        problem::AIE.Problem,
        algorithm::AIE.Algorithm;
        keys,
        iterate,
        kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    remote_results = Dict{eltype(keys), Distributed.Future}()

    return Parallel.DistributedState(; iterate, stopping_criterion_state, remote_results)
end

function AI.initialize_state(
        problem::AIE.Problem,
        algorithm::Parallel.DistributedNestedAlgorithm;
        kwargs...
    )
    return initialize_distributed_state(problem, algorithm; keys = algorithm.keys, kwargs...)
end

function Parallel.DistributedNestedAlgorithm(f::Function, iterable; kwargs...)
    return Parallel.DistributedNestedAlgorithm(; algorithms = map(f, iterable), kwargs...)
end

function AIE.get_subproblem(
        problem::AI.Problem, algorithm::Parallel.DistributedNestedAlgorithm, state::Parallel.DistributedState
    )
    subproblem = problem
    subalgorithm = algorithm.algorithms[state.iteration]

    return subproblem, subalgorithm, state.iterate
end

function AI.step!(
        problem::AI.Problem,
        algorithm::Parallel.DistributedNestedAlgorithm,
        state::Parallel.DistributedState;
        kwargs...
    )

    subproblem, subalgorithm, subiterate = AIE.get_subproblem(problem, algorithm, state)

    # Do whatever should have happened at `step!`, but store the result as a future.

    function solve(subproblem, subalgorithm, iterate)
        rv = AI.solve(subproblem, subalgorithm; iterate)
        return rv
    end

    future = remotecall(solve, algorithm.workers, subproblem, subalgorithm, subiterate)

    AIE.set_substate!(problem, algorithm, state, future)

    return state
end

function AIE.set_substate!(
        ::AIE.Problem,
        algorithm::Parallel.DistributedNestedAlgorithm,
        state::Parallel.DistributedState,
        future::Distributed.Future,
    )
    key = algorithm.keys[state.iteration]

    state.remote_results[key] = future

    return state
end

include("distributedbeliefpropagation.jl")

end # ITensorNetworksNextDistributedExt
