using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs.GraphsExtensions: default_root_vertex, forest_cover, post_order_dfs_edges
using ITensorBase: ITensor, dim

struct BeliefPropagationCache{V, N <: AbstractDataGraph{V}} <:
    AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
network(bp_cache::BeliefPropagationCache) = bp_cache.network

BeliefPropagationCache(network) = BeliefPropagationCache(network, Dictionary())

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

function setmessages!(bpc_dst::BeliefPropagationCache, bpc_src::BeliefPropagationCache, edges)
    ms_dst = messages(bpc_dst)
    for e in edges
        set!(ms_dst, e, message(bpc_src, e))
    end
    return bpc_dst
end

function message(bp_cache::BeliefPropagationCache, edge::AbstractEdge; kwargs...)
    ms = messages(bp_cache)
    return get(() -> default_message(bp_cache, edge; kwargs...), ms, edge)
end

function messages(bp_cache::BeliefPropagationCache, edges::Vector{<:AbstractEdge})
    return [message(bp_cache, e) for e in edges]
end

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

function default_algorithm(
        ::Type{<:Algorithm"contract"}; normalize = true, sequence_alg = "optimal"
    )
    return Algorithm("contract"; normalize, sequence_alg)
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
