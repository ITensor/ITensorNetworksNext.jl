import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

@kwdef struct Sweeping{
        Algorithms <: AbstractVector, StoppingCriterion <: AI.StoppingCriterion,
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
        Algorithms <: AbstractVector, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end
function Sweep(f, nalgorithms::Int; kwargs...)
    return Sweep(; algorithms = f.(1:nalgorithms), kwargs...)
end

struct Region{R, Kwargs <: NamedTuple} <: AIE.NonIterativeAlgorithm
    region::R
    kwargs::Kwargs
end
Region(; region, kwargs...) = Region(region, (; kwargs...))
Region(region; kwargs...) = Region(region, (; kwargs...))
