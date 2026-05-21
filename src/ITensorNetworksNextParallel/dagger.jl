import ..ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI
using ITensorNetworksNext: AbstractBeliefPropagationCache

abstract type Driver end

struct SERIAL <: Driver end
struct DAGGER <: Driver end

@kwdef mutable struct DaggerState{
        Substate, StoppingCriterionState <: AI.StoppingCriterionState, DTask,
    } <: AIE.NestedState
    substate::Substate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
    remote_results::Dict{Int, DTask} = Dict{Int, Any}()
end

function initialize_dagger_state(_problem, _algorithm; _kwargs...)
    throw(
        ErrorException(
            "Package Dagger not loaded. Please install and load the Dagger package."
        )
    )
end

@kwdef struct InParallel{Algorithm <: AI.Algorithm}
    parent::Algorithm

end



function step_dagger!(_problem, _algorithm, _state)
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
