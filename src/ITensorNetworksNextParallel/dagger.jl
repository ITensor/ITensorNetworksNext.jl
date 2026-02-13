import AlgorithmsInterface as AI
import ..ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using ITensorNetworksNext: AbstractBeliefPropagationCache

@kwdef mutable struct DaggerState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState, Chunk, DTask,
    } <: AIE.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
    remote_subiterates::Dict{Int, Chunk} = Dict{Int, Any}()
    remote_results::Dict{Int, DTask} = Dict{Int, Any}()
end

@kwdef struct DaggerNestedAlgorithm{
        ChildAlgorithm <: AIE.Algorithm,
        Algorithms <: AbstractVector{ChildAlgorithm},
        StoppingCriterion <: AI.StoppingCriterion,
        KeyType,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
    workers::Vector{Int}
    keys::Vector{KeyType} = collect(1:length(algorithms))
end

function dagger_algorithm(f::Function, iterable; kwargs...)
    return DaggerNestedAlgorithm(f, iterable; kwargs...)
end

# ================================== belief propagation ================================== #

struct DaggerBeliefPropagationCache{
        V, VD, ED, UC <: AbstractBeliefPropagationCache{V, VD, ED}, Chunks,
    } <: AbstractBeliefPropagationCache{V, VD, ED}
    underlying_cache::UC
    quotient_chunks::Chunks
end
