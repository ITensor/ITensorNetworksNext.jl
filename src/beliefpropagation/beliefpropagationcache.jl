using DiagonalArrays: delta
using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs.GraphsExtensions: default_root_vertex, forest_cover, post_order_dfs_edges
using ITensorBase: ITensor, dim
using TypeParameterAccessors: unwrap_array_type, unwrap_array, parenttype

struct BeliefPropagationCache{V, N <: AbstractDataGraph{V}} <:
    AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
network(bp_cache::BeliefPropagationCache) = bp_cache.network
default_messages() = Dictionary()

BeliefPropagationCache(network) = BeliefPropagationCache(network, default_messages())

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(network(bp_cache)), copy(messages(bp_cache)))
end

function deletemessage!(bp_cache::BeliefPropagationCache, e::AbstractEdge)
    ms = messages(bp_cache)
    delete!(ms, e)
    return bp_cache
end

function setmessage!(bp_cache::BeliefPropagationCache, e::AbstractEdge, message)
    ms = messages(bp_cache)
    set!(ms, e, message)
    return bp_cache
end

function message(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge; kwargs...)
    ms = messages(bp_cache)
    return get(() -> default_message(bp_cache, edge; kwargs...), ms, edge)
end

function messages(bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge})
    return [message(bp_cache, e) for e in edges]
end

default_bp_maxiter(g::AbstractGraph) = is_tree(g) ? 1 : nothing
#Forward onto the network
for f in [
        :(Graphs.vertices),
        :(Graphs.edges),
        :(Graphs.is_tree),
        :(NamedGraphs.GraphsExtensions.boundary_edges),
        :(factors),
        :(default_bp_maxiter),
        :(ITensorNetworksNext.setfactor!),
        :(ITensorNetworksNext.linkinds),
        :(ITensorNetworksNext.underlying_graph),
    ]
    @eval begin
        function $f(bp_cache::BeliefPropagationCache, args...; kwargs...)
            return $f(network(bp_cache), args...; kwargs...)
        end
    end
end

#TODO: Get subgraph working on an ITensorNetwork to overload this directly
function default_bp_edge_sequence(bp_cache::BeliefPropagationCache)
    return forest_cover_edge_sequence(underlying_graph(bp_cache))
end

function factors(tn::AbstractTensorNetwork, vertex)
    return [tn[vertex]]
end

function region_scalar(bp_cache::BeliefPropagationCache, edge::AbstractEdge)
    return (message(bp_cache, edge) * message(bp_cache, reverse(edge)))[]
end

function region_scalar(bp_cache::BeliefPropagationCache, vertex)
    incoming_ms = incoming_messages(bp_cache, vertex)
    state = factors(bp_cache, vertex)
    return (reduce(*, incoming_ms) * reduce(*, state))[]
end

function default_message(bp_cache::BeliefPropagationCache, edge::AbstractEdge)
    return default_message(network(bp_cache), edge::AbstractEdge)
end

function default_message(tn::AbstractTensorNetwork, edge::AbstractEdge)
    t = ITensor(ones(dim.(linkinds(tn, edge))...), linkinds(tn, edge)...)
    #TODO: Get datatype working on tensornetworks so we can support GPU, etc...
    return t
end

#Algorithmic defaults
default_update_alg(bp_cache::BeliefPropagationCache) = "bp"
default_message_update_alg(bp_cache::BeliefPropagationCache) = "contract"
default_normalize(::Algorithm"contract") = true
default_sequence_alg(::Algorithm"contract") = "optimal"
function set_default_kwargs(alg::Algorithm"contract")
    normalize = get(alg, :normalize, default_normalize(alg))
    sequence_alg = get(alg, :sequence_alg, default_sequence_alg(alg))
    return Algorithm("contract"; normalize, sequence_alg)
end
function set_default_kwargs(alg::Algorithm"adapt_update")
    _alg = set_default_kwargs(get(alg, :alg, Algorithm("contract")))
    return Algorithm("adapt_update"; adapt = alg.adapt, alg = _alg)
end
default_verbose(::Algorithm"bp") = false
default_tol(::Algorithm"bp") = nothing
function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCache)
    verbose = get(alg, :verbose, default_verbose(alg))
    maxiter = get(alg, :maxiter, default_bp_maxiter(bp_cache))
    edge_sequence = get(alg, :edge_sequence, default_bp_edge_sequence(bp_cache))
    tol = get(alg, :tol, default_tol(alg))
    message_update_alg = set_default_kwargs(
        get(alg, :message_update_alg, Algorithm(default_message_update_alg(bp_cache)))
    )
    return Algorithm("bp"; verbose, maxiter, edge_sequence, tol, message_update_alg)
end

#TODO: Update message etc should go here...
function updated_message(
        alg::Algorithm"contract", bp_cache::BeliefPropagationCache, edge::AbstractEdge
    )
    vertex = src(edge)
    incoming_ms = incoming_messages(
        bp_cache, vertex; ignore_edges = typeof(edge)[reverse(edge)]
    )
    state = factors(bp_cache, vertex)
    #contract_list = ITensor[incoming_ms; state]
    #sequence = contraction_sequence(contract_list; alg=alg.kwargs.sequence_alg)
    #updated_messages = contract(contract_list; sequence)
    updated_message =
        !isempty(incoming_ms) ? reduce(*, state) * reduce(*, incoming_ms) : reduce(*, state)
    if alg.normalize
        message_norm = LinearAlgebra.norm(updated_message)
        if !iszero(message_norm)
            updated_message /= message_norm
        end
    end
    return updated_message
end

function updated_message(
        bp_cache::BeliefPropagationCache,
        edge::AbstractEdge;
        alg = default_message_update_alg(bpc),
        kwargs...,
    )
    return updated_message(set_default_kwargs(Algorithm(alg; kwargs...)), bp_cache, edge)
end

function update_message!(
        message_update_alg::Algorithm, bp_cache::BeliefPropagationCache, edge::AbstractEdge
    )
    return setmessage!(bp_cache, edge, updated_message(message_update_alg, bp_cache, edge))
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update_iteration(
        alg::Algorithm"bp",
        bpc::AbstractBeliefPropagationCache,
        edges::Vector;
        (update_diff!) = nothing,
    )
    bpc = copy(bpc)
    for e in edges
        prev_message = !isnothing(update_diff!) ? message(bpc, e) : nothing
        update_message!(alg.message_update_alg, bpc, e)
        if !isnothing(update_diff!)
            update_diff![] += message_diff(message(bpc, e), prev_message)
        end
    end
    return bpc
end

"""
Do parallel updates between groups of edges of all message tensors
Currently we send the full message tensor data struct to update for each edge_group. But really we only need the
mts relevant to that group.
"""
function update_iteration(
        alg::Algorithm"bp",
        bpc::AbstractBeliefPropagationCache,
        edge_groups::Vector{<:Vector{<:AbstractEdge}};
        (update_diff!) = nothing,
    )
    new_mts = empty(messages(bpc))
    for edges in edge_groups
        bpc_t = update_iteration(alg.kwargs.message_update_alg, bpc, edges; (update_diff!))
        for e in edges
            set!(new_mts, e, message(bpc_t, e))
        end
    end
    return set_messages(bpc, new_mts)
end

"""
More generic interface for update, with default params
"""
function update(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache)
    compute_error = !isnothing(alg.tol)
    if isnothing(alg.maxiter)
        error("You need to specify a number of iterations for BP!")
    end
    for i in 1:alg.maxiter
        diff = compute_error ? Ref(0.0) : nothing
        bpc = update_iteration(alg, bpc, alg.edge_sequence; (update_diff!) = diff)
        if compute_error && (diff.x / length(alg.edge_sequence)) <= alg.tol
            if alg.verbose
                println("BP converged to desired precision after $i iterations.")
            end
            break
        end
    end
    return bpc
end

function update(bpc::AbstractBeliefPropagationCache; alg = default_update_alg(bpc), kwargs...)
    return update(set_default_kwargs(Algorithm(alg; kwargs...), bpc), bpc)
end

#Edge sequence stuff
function forest_cover_edge_sequence(g::AbstractGraph; root_vertex = default_root_vertex)
    forests = forest_cover(g)
    edges = edgetype(g)[]
    for forest in forests
        trees = [forest[vs] for vs in connected_components(forest)]
        for tree in trees
            tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
            push!(edges, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
        end
    end
    return edges
end