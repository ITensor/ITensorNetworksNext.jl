using DataGraphs: DataGraphs, AbstractDataGraph, edge_data, edge_data_type,
    set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data, vertex_data_type
using Dictionaries: Dictionary, delete!, getindices, set!
using Graphs: AbstractGraph, connected_components, is_directed, is_tree
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, lazy, parenttype
using NamedGraphs.GraphsExtensions: IsDirected, boundary_edges, default_root_vertex,
    directed_graph, forest_cover, in_incident_edges, post_order_dfs_edges, undirected_graph,
    vertextype
using NamedGraphs.PartitionedGraphs: QuotientEdge, QuotientView, quotient_graph
using NamedGraphs: NamedDiGraph, Vertices, convert_vertextype, ordered_vertices,
    parent_graph_indices, position_graph, to_graph_index, vertex_positions

struct MessageCache{T, V} <: AbstractDataGraph{V, Nothing, T}
    messages::Dictionary{NamedEdge{V}, T}
    underlying_graph::NamedDiGraph{V}
    function MessageCache{T, V}(::UndefInitializer, vertices) where {T, V}
        messages = Dictionary{NamedEdge{V}, T}()
        underlying_graph = NamedDiGraph{V}(vertices)
        return new{T, V}(messages, underlying_graph)
    end
end

# A cache that stores sqrt-form messages (in the Vidal-gauge / simple-update
# sense): the entry on each directed edge is the operator that gets contracted
# directly into the state for the balanced gauge — i.e. `√M` rather than the
# "full" message `M`. Structurally identical to `MessageCache`; the apply-
# operator BP path dispatches on the type to use the messages as gauge
# factors directly and skip the sqrt-via-eigh step.
struct SqrtMessageCache{T, V} <: AbstractDataGraph{V, Nothing, T}
    messages::Dictionary{NamedEdge{V}, T}
    underlying_graph::NamedDiGraph{V}
    function SqrtMessageCache{T, V}(::UndefInitializer, vertices) where {T, V}
        messages = Dictionary{NamedEdge{V}, T}()
        underlying_graph = NamedDiGraph{V}(vertices)
        return new{T, V}(messages, underlying_graph)
    end
end

# `MessageCache` and `SqrtMessageCache` are sibling concrete types: the storage
# and graph structure are identical, only the semantic interpretation of the
# message values differs. Shared methods are emitted per-type via this loop
# rather than via a shared abstract supertype. Once
# `DataGraphs.AbstractEdgeDataGraph` (DataGraphs.jl#121) lands, both can
# subtype that and most of this loop can fall away.
for Cache in (:MessageCache, :SqrtMessageCache)
    @eval begin
        # ============================ constructors ===================================== #

        function $Cache{T}(::UndefInitializer, vertices) where {T}
            return $Cache{T, eltype(vertices)}(undef, vertices)
        end

        $Cache(messages) = $Cache{valtype(messages)}(messages)

        function $Cache{T}(messages) where {T}
            V = vertextype(keytype(messages))
            return $Cache{T, V}(messages)
        end

        # `messages` is any iterable data structure, where `keys(messages)`
        # are edges and the values are the messages on those edges.
        function $Cache{T, V}(messages) where {T, V}
            edges = keys(messages)
            vertices = union(src.(edges), dst.(edges))
            cache = $Cache{T, V}(undef, vertices)
            add_edges!(cache.underlying_graph, edges)
            copyto!(cache, messages)
            return cache
        end

        Base.copy(cache::$Cache) = $Cache(copy(cache.messages))

        # ============================ key/val types ==================================== #

        Base.keytype(c::$Cache) = keytype(typeof(c))
        Base.keytype(::Type{<:$Cache{T, V}}) where {T, V} = NamedEdge{V}
        Base.valtype(c::$Cache) = valtype(typeof(c))
        Base.valtype(::Type{<:$Cache{T}}) where {T} = T
        Base.keys(cache::$Cache) = edges(cache)

        # ============================ NamedGraphs interface ============================ #

        function NamedGraphs.add_edge!(c::$Cache, edge)
            add_edge!(c.underlying_graph, edge)
            return c
        end

        function NamedGraphs.rem_edge!(c::$Cache, edge)
            delete!(c.messages, to_graph_index(c, edge))
            rem_edge!(c.underlying_graph, edge)
            return c
        end

        function NamedGraphs.induced_subgraph_from_vertices(cache::$Cache, subvertices)
            # TODO: once we have `subgraph_edges` in `NamedGraphs`, simplify this.
            underlying_subgraph, vlist =
                Graphs.induced_subgraph(cache.underlying_graph, subvertices)
            assigned = v -> isassigned(cache, v)
            assigned_subedges = Iterators.filter(assigned, edges(underlying_subgraph))
            messages = getindices(cache.messages, Indices(assigned_subedges))
            return $Cache(messages), vlist
        end

        # ============================ DataGraphs interface ============================= #

        DataGraphs.underlying_graph(cache::$Cache) = cache.underlying_graph
        DataGraphs.is_vertex_assigned(::$Cache, _) = false
        DataGraphs.is_edge_assigned(c::$Cache, edge) = haskey(c.messages, edge)

        function DataGraphs.get_edge_data(c::$Cache, edge::AbstractEdge)
            return c.messages[edge]
        end
        function DataGraphs.set_edge_data!(c::$Cache, val, edge)
            return set!(c.messages, edge, val)
        end

        # ============================ equality ========================================= #

        function Base.:(==)(c1::$Cache, c2::$Cache)
            return c1.underlying_graph == c2.underlying_graph && c1.messages == c2.messages
        end

        # ============================ copyto! ========================================== #

        # see: copyto!(dest, src) for analogous behaviour to 2 argument method
        # see: copyto!(dest, Rdest::CartesianIndices, src, Rsrc::CartesianIndices)
        # for analogous behaviour to 3 argument method.
        # TODO: these can be made generic for `AbstractDataGraph` in `DataGraphs.jl`.
        function Base.copyto!(
                cache_dst::$Cache, cache_src::AbstractDataGraph, inds = nothing
            )
            copyto!_messagecache(cache_dst, edge_data(cache_src), inds)
            return cache_dst
        end

        function Base.copyto!(
                cache_dst::$Cache, dictionary_src::Dictionary, inds = nothing
            )
            copyto!_messagecache(cache_dst, dictionary_src, inds)
            return cache_dst
        end

        function Base.copyto!(
                cache_dst::$Cache, dict_src::Dict, inds = keys(dict_src)
            )
            for key in inds
                cache_dst[key] = dict_src[key]
            end
            return cache_dst
        end

        # ============================ printing ========================================= #

        # TODO: This is the definition for the proposed `DataGraphs.AbstractEdgeDataGraph`.
        function Base.show(io::IO, mime::MIME"text/plain", graph::$Cache)
            println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
            show(io, mime, vertices(graph))
            println(io, "\n")
            println(io, "and $(ne(graph)) edge(s):")
            for e in edges(graph)
                show(io, mime, e)
                println(io)
            end
            println(io)
            println(io, "with edge data:")
            show(io, mime, edge_data(graph))
            return nothing
        end

        Base.show(io::IO, graph::$Cache) = show(io, MIME"text/plain"(), graph)
    end
end

messagecache(pairs) = MessageCache(Dict(pairs))
messagecache(f, edges) = messagecache(edge => f(edge) for edge in edges)

sqrtmessagecache(pairs) = SqrtMessageCache(Dict(pairs))
sqrtmessagecache(f, edges) = sqrtmessagecache(edge => f(edge) for edge in edges)

function copyto!_messagecache(cache_dst, cache_src, inds = nothing)
    inds = isnothing(inds) ? Indices(keys(cache_src)) : Indices(inds)
    view(edge_data(cache_dst), inds) .= view(cache_src, inds)
    return cache_dst
end

# ===================================== contraction ====================================== #

function incoming_messages(cache::AbstractGraph, pair::Pair)
    edge = to_graph_index(cache, pair)
    return incoming_messages(cache, edge)
end
function incoming_messages(cache::AbstractGraph, edge::AbstractEdge)
    inds = Indices(in_incident_edges(cache, src(edge)))
    return getindices(cache, filter(e -> e != reverse(edge), inds))
end

# TODO: maybe this should be defined in `DataGraphs`.
function incoming_edge_data(cache::AbstractGraph, vertices)
    inds = Indices(boundary_edges(cache, vertices; dir = :in))
    return getindices(cache, inds)
end

function vertex_scalar(factors, messages, vertex; kwargs...)
    in_messages = incoming_edge_data(messages, [vertex])
    tensors = vcat([factors[vertex]], collect(in_messages))
    return contract_network(tensors; kwargs...)[]
end

vertex_scalars(factors, messages) = vertex_scalars(factors, messages, keys(factors))
function vertex_scalars(factors::AbstractGraph, messages)
    return vertex_scalars(factors, messages, vertices(factors))
end
function vertex_scalars(factors, messages, vertices)
    return map(v -> vertex_scalar(factors, messages, v), vertices)
end

function edge_scalar(cache, edge; kwargs...)
    m1 = cache[edge]
    m2 = cache[reverse(edge)]
    return contract_network([m1, m2]; kwargs...)[]
end

edge_scalars(cache) = edge_scalars(cache, keys(cache))

function edge_scalars(cache, edges)
    processed = Set{eltype(edges)}()

    T = Base.promote_op(edge_scalar, typeof(cache), eltype(edges))

    scalars = T[]

    # Ignore repeated edges and their reverses.
    for e in edges
        if e in processed || reverse(e) in processed
            continue
        end
        push!(processed, e)
        push!(scalars, edge_scalar(cache, e))
    end

    return scalars
end

function region_scalar(factors, messages, region)
    return mapreduce(vertex -> vertex_scalar(factors, messages, vertex), *, region)
end

# We need a graph structure here, so assume `factors` is a graph.
function bethe_free_energy(factors, messages)
    numerator_terms = vertex_scalars(factors, messages)
    denominator_terms = edge_scalars(messages)

    if any(t -> real(t) < 0, numerator_terms)
        numerator_terms = complex.(numerator_terms)
    end
    if any(t -> real(t) < 0, denominator_terms)
        denominator_terms = complex.(denominator_terms)
    end

    if any(iszero, denominator_terms)
        return -Inf
    end

    return sum(log.(numerator_terms)) - sum(log.(denominator_terms))
end

# TODO: This needs to go in NamedGraphs.GraphsExtensions
function forest_cover_edge_sequence(gi::AbstractGraph; root_vertex = default_root_vertex)
    # All we care about are the edges so the type of the graph doesnt matter
    g = similar_graph(NamedGraph, vertices(gi))
    add_edges!(g, edges(gi))
    forests = forest_cover(g)
    rv = edgetype(g)[]
    for forest in forests
        trees = [forest[Vertices(vs)] for vs in connected_components(forest)]
        for tree in trees
            tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
            push!(rv, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
        end
    end
    return rv
end
