import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

maybe_fill(value, len::Int) = fill(value, len)
function maybe_fill(v::AbstractVector, len::Int)
    @assert length(v) == len
    return v
end

function dmrg_sweep(operator, algorithm, state)
    problem = select_problem(dmrg_sweep, operator, algorithm, state)
    return AI.solve(problem, algorithm; iterate = state).iterate
end
function dmrg_sweep(operator, state; kwargs...)
    algorithm = select_algorithm(dmrg_sweep, operator, state; kwargs...)
    return dmrg_sweep(operator, algorithm, state)
end

function select_problem(::typeof(dmrg_sweep), operator, algorithm, state)
    return EigenProblem(operator)
end
function select_algorithm(::typeof(dmrg_sweep), operator, state; regions, region_kwargs)
    region_kwargs′ = maybe_fill(region_kwargs, length(regions))
    return Sweep(length(regions)) do i
        return Returns(Region(regions[i]; region_kwargs′[i]...))
    end
end

function dmrg(operator, algorithm, state)
    problem = select_problem(dmrg, operator, algorithm, state)
    return AI.solve(problem, algorithm; iterate = state).iterate
end
function dmrg(operator, state; kwargs...)
    algorithm = select_algorithm(dmrg, operator, state; kwargs...)
    return dmrg(operator, algorithm, state)
end

function select_problem(::typeof(dmrg), operator, algorithm, state)
    return EigenProblem(operator)
end
function select_algorithm(::typeof(dmrg), operator, state; nsweeps, regions, region_kwargs)
    region_kwargs′ = maybe_fill(region_kwargs, nsweeps)
    return Sweeping(nsweeps) do i
        return select_algorithm(
            dmrg_sweep, operator, state;
            regions, region_kwargs = region_kwargs′[i],
        )
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

function AI.step!(problem::EigenProblem, algorithm::Region, state::AIE.State; kwargs...)
    return error("Region step for EigenProblem not implemented.")
end
