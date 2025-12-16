import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

@kwdef struct Sweeping{
        Algorithms <: AbstractVector{<:AI.Algorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end
function Sweeping(f::Function, nalgorithms::Int; kwargs...)
    return Sweeping(; algorithms = f.(1:nalgorithms), kwargs...)
end

#=
    Sweep(regions::AbsractVector, region_kwargs::Function, iteration::Int = 0)
    Sweep(regions::AbsractVector, region_kwargs::NamedTuple, iteration::Int = 0)

The "algorithm" for performing a single sweep over a list of regions. It also
stores a function that takes the problem, algorithm, and state (tensor network, current
region, etc.) and returns keyword arguments for performing the region update on the
current region. For simplicity, it also accepts a `NamedTuple` of keyword arguments
which is converted into a function that always returns the same keyword arguments
for an region.
=#
@kwdef struct Sweep{
        RegionAlgorithms <: AbstractVector, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.Algorithm
    region_algorithms::RegionAlgorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(region_algorithms))
end
function Sweep(f, nalgorithms::Int; kwargs...)
    region_algorithms = to_region_algorithm.(f.(1:nalgorithms))
    return Sweep(; region_algorithms, kwargs...)
end
to_region_algorithm(algorithm::Function) = algorithm
to_region_algorithm(algorithm) = Returns(region_algorithm(algorithm))

AIE.max_iterations(algorithm::Sweep) = length(algorithm.algorithms)

abstract type RegionAlgorithm end
region_algorithm(algorithm::RegionAlgorithm) = algorithm
region_algorithm(algorithm::NamedTuple) = Region(; algorithm...)

struct Region{R, Kwargs <: NamedTuple} <: RegionAlgorithm
    region::R
    kwargs::Kwargs
end
function Region(; region, kwargs...)
    return Region(region, (; kwargs...))
end
function Region(region; kwargs...)
    return Region(region, (; kwargs...))
end
