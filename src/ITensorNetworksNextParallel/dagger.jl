import ..ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using Dictionaries: Dictionary
using ITensorNetworksNext: AbstractBeliefPropagationCache

@kwdef mutable struct DaggerState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState, DTask,
    } <: AIE.State
    iterate::Iterate # DaggerBeliefPropagationCache
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
    remote_results::Dictionary{Int, DTask} = Dict{Int, Any}()
end

function initialize_dagger_state(problem, algorithm; kwargs...)
    throw(
        ErrorException(
            "Package Dagger not loaded. Please install and load the Dagger package."
        )
    )
end

@kwdef struct DaggerNestedAlgorithm{
        ChildAlgorithm <: AIE.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: ParallelAlgorithm{ChildAlgorithm}
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end

function dagger_algorithm(f, iterable; kwargs...)
    throw(
        ErrorException(
            "Package Dagger not loaded. Please install and load the Dagger package."
        )
    )
end

# ================================== belief propagation ================================== #

struct DaggerBeliefPropagationCache{
        V, VD, ED, UC <: AbstractBeliefPropagationCache{V, VD, ED}, Chunks,
    } <: AbstractBeliefPropagationCache{V, VD, ED}
    underlying_cache::UC
    quotient_chunks::Chunks
end
