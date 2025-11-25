using Graphs: SimpleGraph, vertices, edges, has_edge
using NamedGraphs: AbstractNamedGraph, position_graph
using NamedGraphs.GraphsExtensions: add_edges!, partition_vertices
using NamedGraphs.OrderedDictionaries: OrderedDictionary, OrderedIndices

abstract type AbstractBeliefPropagationProblem{Alg} <: AbstractProblem end

mutable struct BeliefPropagationProblem{Alg, Cache} <: AbstractBeliefPropagationProblem{Alg}
    const alg::Alg
    const cache::Cache
    diff::Union{Nothing, Float64}
end

BeliefPropagationProblem(alg, cache) = BeliefPropagationProblem(alg, cache, nothing)

function default_algorithm(
        ::Type{<:Algorithm"bp"},
        bpc;
        verbose = false,
        tol = nothing,
        edge_sequence = forest_cover_edge_sequence(bpc),
        message_update_alg = default_algorithm(Algorithm"contract"),
        maxiter = is_tree(bpc) ? 1 : nothing,
    )
    return Algorithm("bp"; verbose, tol, edge_sequence, message_update_alg, maxiter)
end

function region_plan(prob::BeliefPropagationProblem{<:Algorithm"bp"}; sweep_kwargs...)
    edges = prob.alg.edge_sequence

    plan = map(edges) do e
        return e => (; sweep_kwargs...)
    end

    return plan
end

function compute!(iter::RegionIterator{<:BeliefPropagationProblem{<:Algorithm"bp"}})
    prob = iter.problem

    edge, _ = current_region_plan(iter)
    new_message = updated_message(prob.alg.message_update_alg, prob.cache, edge)
    setmessage!(prob.cache, edge, new_message)

    return iter
end

default_message(alg, network, edge) = default_message(typeof(alg), network, edge)

default_message(::Type{<:Algorithm}, network, edge) = not_implemented()
function default_message(::Type{<:Algorithm"bp"}, network, edge)

    #TODO: Get datatype working on tensornetworks so we can support GPU, etc...
    links = linkinds(network, edge)
    data = ones(Tuple(links))
    return data
end

updated_message(alg, bpc, edge) = not_implemented()
function updated_message(alg::Algorithm"contract", bpc, edge)
    vertex = src(edge)

    incoming_ms = incoming_messages(
        bpc, vertex; ignore_edges = typeof(edge)[reverse(edge)]
    )

    updated_message = contract_messages(alg.contraction_alg, factors(bpc, vertex), incoming_ms)

    if alg.normalize
        message_norm = LinearAlgebra.norm(updated_message)
        if !iszero(message_norm)
            updated_message /= message_norm
        end
    end
    return updated_message
end

contract_messages(alg, factors, messages) = not_implemented()
function contract_messages(
        alg,
        factors::Vector{<:AbstractArray},
        messages::Vector{<:AbstractArray},
    )
    return contract_network(alg, vcat(factors, messages))
end

function default_algorithm(
        ::Type{<:Algorithm"contract"}; normalize = true, contraction_alg = Algorithm("exact")
    )
    return Algorithm("contract"; normalize, contraction_alg)
end
function default_algorithm(
        ::Type{<:Algorithm"adapt_update"}; adapt, alg = default_algorithm(Algorithm"contract")
    )
    return Algorithm("adapt_update"; adapt, alg)
end

function update_message!(
        message_update_alg::Algorithm, bpc::BeliefPropagationCache, edge::AbstractEdge
    )
    return setmessage!(bpc, edge, updated_message(message_update_alg, bpc, edge))
end

function update(bpc::AbstractBeliefPropagationCache; kwargs...)
    return update(default_algorithm(Algorithm"bp", bpc; kwargs...), bpc)
end

function update(alg, bpc)
    compute_error = !isnothing(alg.tol)

    diff = compute_error ? 0.0 : nothing

    prob = BeliefPropagationProblem(alg, bpc, diff)

    iter = SweepIterator(prob, alg.maxiter; compute_error)

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
