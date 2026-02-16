import .AlgorithmsInterfaceExtensions as AIE
import AlgorithmsInterface as AI

function dmrg(operator, algorithm, state)
    problem = EigenProblem(operator)
    return AI.solve(problem, algorithm; iterate = state).iterate
end
function dmrg(operator, state; kwargs...)
    problem = EigenProblem(operator)
    algorithm = select_algorithm(dmrg, operator, state; kwargs...)
    return AI.solve(problem, algorithm; iterate = state).iterate
end

# TODO: Allow specifying the region algorithm type?
function select_algorithm(::typeof(dmrg), operator, state; nsweeps, regions, kwargs...)
    extended_kwargs = extend_columns((; kwargs...), nsweeps)
    region_kwargs = rows(extended_kwargs)
    return AIE.nested_algorithm(nsweeps) do i
        return AIE.nested_algorithm(length(regions)) do j
            return EigsolveRegion(regions[j]; region_kwargs[i]...)
        end
    end
end
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
