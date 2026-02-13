import AlgorithmsInterface as AI
import ..ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE

using ..ITensorNetworksNext: AbstractBeliefPropagationCache

@kwdef mutable struct DistributedState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState, Future, KeyType,
    } <: AIE.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
    remote_results::Dict{KeyType, Future} = Dict{Int, Any}()
end

@kwdef struct DistributedNestedAlgorithm{
        ChildAlgorithm <: AIE.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
        WorkerPool,
        KeyType,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
    workers::WorkerPool
    keys::Vector{KeyType} = collect(1:length(algorithms))
end

function distributed_algorithm(f::Function, iterable; kwargs...)
    return DistributedNestedAlgorithm(f, iterable; kwargs...)
end

# ================================== belief propagation ================================== #

struct DistributedBeliefPropagationCache{
        V, VD, ED, UC <: AbstractBeliefPropagationCache{V, VD, ED},
    } <: AbstractBeliefPropagationCache{V, VD, ED}
    underlying_cache::UC
end
