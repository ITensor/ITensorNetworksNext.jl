import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

#=
    EigenProblem(operator)

Represents the problem we are trying to solve and minimal algorithm-independent
information, so for an eigenproblem it is the operator we want the eigenvector of.
=#
struct EigenProblem{Operator} <: AIE.Problem
    operator::Operator
end

struct EigsolveRegion{R, Kwargs <: NamedTuple} <: AIE.NonIterativeAlgorithm
    region::R
    kwargs::Kwargs
end
EigsolveRegion(region; kwargs...) = EigsolveRegion(region, (; kwargs...))

function AI.solve!(
        problem::EigenProblem, algorithm::EigsolveRegion, state::AIE.State; kwargs...
    )
    return error("EigsolveRegion step for EigenProblem not implemented yet.")
end

maybe_fill(value, len::Int) = fill(value, len)
function maybe_fill(v::AbstractVector, len::Int)
    @assert length(v) == len
    return v
end

function dmrg(operator, algorithm, state)
    problem = EigenProblem(operator)
    return AI.solve(problem, algorithm; iterate = state).iterate
end
function dmrg(operator, state; kwargs...)
    problem = EigenProblem(operator)
    algorithm = select_algorithm(dmrg, operator, state; kwargs...)
    return AI.solve(problem, algorithm; iterate = state).iterate
end

function repeat_last(v::AbstractVector, len::Int)
    length(v) ≥ len && return v
    return [v; fill(v[end], len - length(v))]
end
repeat_last(v, len::Int) = fill(v, len)
function extend_columns(nt::NamedTuple, len::Int)
    return NamedTuple{keys(nt)}(map(v -> repeat_last(v, len), values(nt)))
end
function eachrow(nt::NamedTuple, len::Int)
    return [NamedTuple{keys(nt)}(map(v -> v[i], values(nt))) for i in 1:len]
end

function select_algorithm(::typeof(dmrg), operator, state; nsweeps, regions, kwargs...)
    extended_kwargs = extend_columns((; kwargs...), nsweeps)
    region_kwargs = eachrow(extended_kwargs, nsweeps)
    return AIE.nested_algorithm(nsweeps) do i
        return AIE.nested_algorithm(length(regions)) do j
            return EigsolveRegion(regions[j]; region_kwargs[i]...)
        end
    end
end
