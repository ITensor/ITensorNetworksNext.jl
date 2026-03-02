module ITensorNetworksNextParallel

import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE

abstract type ParallelAlgorithm{Child} <: AIE.NestedAlgorithm{Child} end
const IterativeParallelAlgorithm{Child <: ParallelAlgorithm} = AIE.NestedAlgorithm{Child}

"""
    get_subiterate(subproblem::AI.Problem, subalgorithm::AI.Algorithm, state::AI.State)

For a given `subproblem` and `subalgorithm` of a parent nested algorithm,
derive (from the parent state `state`) the iterate to be used in the associated sub state.
The returned value of this function is then pass to a remote call of `initialize_state`.
"""
get_subiterate(::AI.Problem, ::AI.Algorithm, state::AI.State) = state.iterate

finalize_state!(::AI.Problem, ::AI.Algorithm, state::AI.State) = state

function AI.is_finished!(
        problem::AI.Problem,
        algorithm::IterativeParallelAlgorithm,
        state::AI.State
    )
    c = algorithm.stopping_criterion
    st = state.stopping_criterion_state

    isfinished = AI.is_finished!(problem, algorithm, state, c, st)

    if isfinished
        finalize_state!(problem, algorithm, state)
    end

    return isfinished
end

include("dagger.jl")

end # ITensorNetworksNextParallel
