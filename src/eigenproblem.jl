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

function AI.step!(problem::EigenProblem, algorithm::Sweep, state::AI.State; kwargs...)
    iterate = solve_region!!(
        problem, algorithm.region_algorithms[state.iteration](state.iterate), state.iterate
    )
    state.iterate = iterate
    return state
end

# extract!, update!, insert! for the region.
function solve_region!!(problem::EigenProblem, algorithm::RegionAlgorithm, state)
    operator = problem.operator
    region = algorithm.region
    region_kwargs = algorithm.kwargs

    #=
    # Reduce the `operator` and state `x` onto the region `region`,
    # and call `eigsolve` on the reduced operator and state using the
    # keyword arguments determined from `region_kwargs`.
    operator_region = reduced_operator(operator, x, region)
    x_region = reduced_state(x, region)
    x_region′ = eigsolve(operator_region, x_region; region_kwargs.update...)
    x′ = insert(x, region, x_region′; region_kwargs.insert...)
    state.state = x′
    =#

    # Dummy update for demonstration purposes.
    state′ = "region = $region" *
        ", update_kwargs = $(region_kwargs.update)" *
        ", insert_kwargs = $(region_kwargs.insert)"
    state = [state; [state′]]

    return state
end
