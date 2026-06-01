module ITensorNetworksNextParallel

import ..ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI

abstract type AbstractParallelizationStrategy end

function default_workers end
function initialize_parallel_state end

@kwdef struct Parallelized{Strategy, Workers, Algorithm <: AI.Algorithm} <: AI.Algorithm
    parent::Algorithm
    strategy::Strategy
    workers::Workers = default_workers(parent, strategy)
end

function Base.getproperty(algorithm::Parallelized, name::Symbol)
    if name in (:parent, :strategy, :workers)
        return getfield(algorithm, name)
    end
    return getproperty(getfield(algorithm, :parent), name)
end

function AI.initialize_state(problem::AI.Problem, algorithm::Parallelized; kwargs...)
    return initialize_parallel_state(
        problem,
        algorithm.parent,
        algorithm.strategy;
        kwargs...
    )
end

# ====================================== Dagger.jl ======================================= #

abstract type AbstractDaggerStrategy <: AbstractParallelizationStrategy end
struct GenericDaggerStrategy <: AbstractDaggerStrategy end

function initialize_parallel_state(
        _problem,
        _algorithm,
        strategy::AbstractDaggerStrategy;
        _kwargs...
    )
    throw(
        ArgumentError(
            "package Dagger.jl not loaded; please install and load Dagger.jl to use \
            strategy of type $(typeof(strategy))."
        )
    )
end

function default_workers(algorithm, strategy::AbstractDaggerStrategy)
    @warn(
        "package Dagger.jl may not be loaded; please install and load Dagger.jl to use \
        strategy of type `$(typeof(strategy))`"
    )
    throw(MethodError(default_workers, (algorithm, strategy)))
end

end # ITensorNetworksNextParallel
