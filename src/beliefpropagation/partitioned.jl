@kwdef struct PartitionedBeliefPropagationSweep{
        Subfactors,
        MessageUpdateAlgorithm,
    } <: AIE.NestedAlgorithm
    partitioned_factors::Vector{Subfactors}
    message_update_algorithm::MessageUpdateAlgorithm
end

function Base.getproperty(alg::PartitionedBeliefPropagationSweep, name::Symbol)
    if name === :stopping_criterion
        return AI.StopAfterIteration(length(alg.partitioned_factors))
    end
    return getfield(alg, name)
end

function AIE.initialize_subsolve(
        problem::BeliefPropagationSweepProblem,
        algorithm::PartitionedBeliefPropagationSweep,
        state::AI.State
    )
    subvertices = algorithm.partitioned_factors[state.iteration]
    cache = state.iterate

    incoming_edges = boundary_edges(problem.factors, subvertices; dir = :in)

    factors = subgraph(problem.factors, subvertices)
    iterate = subgraph(cache, subvertices)


    for edge in incoming_edges
        add_vertex!(iterate, src(edge))
        add_vertex!(iterate, dst(edge))

        iterate[edge] = cache[edge]
        iterate[reverse(edge)] = cache[reverse(edge)]
    end

    # Don't want to update the incoming messages.
    subedges = setdiff(edges(iterate), incoming_edges)

    subproblem = BeliefPropagationSweepProblem(factors, subedges)

    subalgorithm = BeliefPropagationSweepAlgorithm(;
        message_update_algorithm = algorithm.message_update_algorithm,
        stopping_criterion = AI.StopAfterIteration(length(subedges))
    )

    substate = AI.initialize_state(subproblem, subalgorithm; iterate)

    return subproblem, subalgorithm, substate
end

function AIE.finalize_substate!(
        subproblem::BeliefPropagationSweepProblem,
        _subalgorithm::BeliefPropagationSweepAlgorithm,
        substate::AI.State,
        state::AI.State
    )
    subcache = substate.iterate
    subedges = subproblem.edges

    for edge in subedges
        state.iterate[edge] = subcache[edge]
    end

    return state
end

function beliefpropagation(
        factors, messages, partitions;
        edges = default_beliefpropagation_edges(factors),
        stopping_criterion = nothing,
        message_update_algorithm = nothing,
        sweep_algorithm = nothing
    )
    problem = BeliefPropagationProblem(factors)
    cache = MessageCache(messages)

    # No concrete `edge` value here, so the args tuple uses `edgetype(factors)`.
    message_update_algorithm = AIE.select_algorithm(
        message_update!,
        message_update_algorithm,
        Tuple{typeof(cache), typeof(factors), edgetype(factors)}
    )

    if isnothing(sweep_algorithm)
        sweep_algorithm = PartitionedBeliefPropagationSweep(;
            partitioned_factors = partitions,
            message_update_algorithm
        )
    end
    stopping_criterion = select_beliefpropagation_stopping_criterion(stopping_criterion)
    algorithm = BeliefPropagationAlgorithm(;
        edges,
        subalgorithm = sweep_algorithm,
        stopping_criterion
    )

    return AI.solve(problem, algorithm; iterate = cache) # -> typeof(cache)
end

# function message_update!(algorithm::KrylovMessageUpdate, cache, factors, path)
#     function f(messages)
#         temp_cache = copy(cache) # shallow copy.
#
#         for (edge, message) in zip(path, messages)
#             temp_cache[edge] = message
#         end
#
#         for (i, edge) in enumerate(path)
#             message_update!(
#                 algorithm.message_update_algorithm,
#                 temp_cache,
#                 factors,
#                 edge
#             )
#             messages[i] = temp_cache[edge]
#         end
#
#         return messages
#     end
#
#     # do eigsolve step.
#
#     return cache
# end
