mutable struct BeliefPropagationProblem{V, Cache <: AbstractBeliefPropagationCache{V}} <:
    AbstractProblem
    const cache::Cache
    diff::Union{Nothing, Float64}
end

function default_algorithm(
        ::Type{<:Algorithm"bp"},
        bpc::BeliefPropagationCache;
        verbose = false,
        tol = nothing,
        edge_sequence = forest_cover_edge_sequence(network(bpc)),
        message_update_alg = default_algorithm(Algorithm"contract"),
        maxiter = is_tree(bpc) ? 1 : nothing,
    )
    return Algorithm("bp"; verbose, tol, edge_sequence, message_update_alg, maxiter)
end

function compute!(iter::RegionIterator{<:BeliefPropagationProblem})
    prob = iter.problem

    edge_group, kwargs = current_region_plan(iter)

    new_message_tensors = map(edge_group) do edge
        old_message = message(prob.cache, edge)

        new_message = updated_message(kwargs.message_update_alg, prob.cache, edge)

        if !isnothing(prob.diff)
            # TODO: Define `message_diff`
            prob.diff += message_diff(new_message, old_message)
        end

        return new_message
    end

    foreach(edge_group, new_message_tensors) do edge, new_message
        setmessage!(prob.cache, edge, new_message)
    end

    return iter
end

function region_plan(
        prob::BeliefPropagationProblem; root_vertex = default_root_vertex, sweep_kwargs...
    )

    edges = forest_cover_edge_sequence(network(prob.cache); root_vertex)

    plan = map(edges) do e
        return [e] => (; sweep_kwargs...)
    end

    return plan
end

function update(bpc::AbstractBeliefPropagationCache; kwargs...)
    return update(default_algorithm(Algorithm"bp", bpc; kwargs...), bpc)
end
function update(alg::Algorithm"bp", bpc)
    compute_error = !isnothing(alg.tol)

    diff = compute_error ? 0.0 : nothing

    prob = BeliefPropagationProblem(bpc, diff)

    iter = SweepIterator(prob, alg.maxiter; compute_error, getfield(alg, :kwargs)...)

    for _ in iter
        if compute_error && prob.diff <= alg.tol
            break
        end
    end

    if alg.verbose && compute_error
        if prob.diff <= alg.tol
            println("BP converged to desired precision after $(iter.which_sweep) iterations.")
        else
            println(
                "BP failed to converge to precision $(alg.tol), got $(prob.diff) after $(iter.which_sweep) iterations",
            )
        end
    end

    return bpc
end
