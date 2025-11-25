using Adapt: Adapt, adapt, adapt_structure
using BackendSelection: @Algorithm_str, Algorithm
using DataGraphs: DataGraphs, AbstractDataGraph, edge_data, underlying_graph,
    underlying_graph_type, vertex_data
using Dictionaries: Dictionary
using Graphs: Graphs, AbstractEdge, AbstractGraph, Graph, add_edge!, add_vertex!,
    bfs_tree, center, dst, edges, edgetype, ne, neighbors, nv, rem_edge!, src, vertices
using LinearAlgebra: LinearAlgebra, factorize
using MacroTools: @capture
using NamedDimsArrays: dimnames, inds
using NamedGraphs: NamedGraphs, NamedGraph, not_implemented, steiner_tree
using NamedGraphs.OrdinalIndexing: OrdinalSuffixedInteger
using NamedGraphs.GraphsExtensions:
    ⊔,
    directed_graph,
    incident_edges,
    rem_edges!,
    rename_vertices,
    vertextype
using SplitApplyCombine: flatten
using NamedGraphs.SimilarType: similar_type

abstract type AbstractTensorNetwork{V, VD} <: AbstractDataGraph{V, VD, Nothing} end

# Need to be careful about removing edges from tensor networks in case there is a bond
Graphs.rem_edge!(::AbstractTensorNetwork, edge) = not_implemented()

DataGraphs.edge_data_eltype(::Type{<:AbstractTensorNetwork}) = not_implemented()

# Graphs.jl overloads
function Graphs.weights(graph::AbstractTensorNetwork)
    V = vertextype(graph)
    es = Tuple.(edges(graph))
    ws = Dictionary{Tuple{V, V}, Float64}(es, undef)
    for e in edges(graph)
        w = log2(dim(commoninds(graph, e)))
        ws[(src(e), dst(e))] = w
    end
    return ws
end

# Copy
Base.copy(::AbstractTensorNetwork) = not_implemented()

# Iteration
Base.iterate(tn::AbstractTensorNetwork, args...) = iterate(vertex_data(tn), args...)
Base.keys(tn::AbstractTensorNetwork) = vertices(tn)

# TODO: This contrasts with the `DataGraphs.AbstractDataGraph` definition,
# where it is defined as the `vertextype`. Does that cause problems or should it be changed?
Base.eltype(tn::AbstractTensorNetwork) = eltype(vertex_data(tn))

# Overload if needed
Graphs.is_directed(::Type{<:AbstractTensorNetwork}) = false

# AbstractDataGraphs overloads
DataGraphs.vertex_data(::AbstractTensorNetwork) = not_implemented()
DataGraphs.edge_data(::AbstractTensorNetwork) = not_implemented()

DataGraphs.underlying_graph(::AbstractTensorNetwork) = not_implemented()
function NamedGraphs.vertex_positions(tn::AbstractTensorNetwork)
    return NamedGraphs.vertex_positions(underlying_graph(tn))
end
function NamedGraphs.ordered_vertices(tn::AbstractTensorNetwork)
    return NamedGraphs.ordered_vertices(underlying_graph(tn))
end

function Adapt.adapt_structure(to, tn::AbstractTensorNetwork)
    # TODO: Define and use:
    #
    # @preserve_graph map_vertex_data(adapt(to), tn)
    #
    # or just:
    #
    # @preserve_graph map(adapt(to), tn)
    return map_vertex_data_preserve_graph(adapt(to), tn)
end

linkinds(tn::AbstractGraph, edge::Pair) = linkinds(tn, edgetype(tn)(edge))
linkinds(tn::AbstractGraph, edge::AbstractEdge) = inds(tn[src(edge)]) ∩ inds(tn[dst(edge)])

function linkaxes(tn::AbstractGraph, edge::Pair)
    return linkaxes(tn, edgetype(tn)(edge))
end
function linkaxes(tn::AbstractGraph, edge::AbstractEdge)
    return axes(tn[src(edge)]) ∩ axes(tn[dst(edge)])
end
function linknames(tn::AbstractGraph, edge::Pair)
    return linknames(tn, edgetype(tn)(edge))
end
function linknames(tn::AbstractGraph, edge::AbstractEdge)
    return dimnames(tn[src(edge)]) ∩ dimnames(tn[dst(edge)])
end

function siteinds(tn::AbstractGraph, v)
    s = inds(tn[v])
    for v′ in neighbors(tn, v)
        s = setdiff(s, inds(tn[v′]))
    end
    return s
end
function siteaxes(tn::AbstractGraph, edge::AbstractEdge)
    s = axes(tn[src(edge)]) ∩ axes(tn[dst(edge)])
    for v′ in neighbors(tn, v)
        s = setdiff(s, axes(tn[v′]))
    end
    return s
end
function sitenames(tn::AbstractGraph, edge::AbstractEdge)
    s = dimnames(tn[src(edge)]) ∩ dimnames(tn[dst(edge)])
    for v′ in neighbors(tn, v)
        s = setdiff(s, dimnames(tn[v′]))
    end
    return s
end

function setindex_preserve_graph!(tn::AbstractGraph, value, vertex)
    set!(vertex_data(tn), vertex, value)
    return tn
end

# TODO: Move to `BaseExtensions` module.
function is_setindex!_expr(expr::Expr)
    return is_assignment_expr(expr) && is_getindex_expr(first(expr.args))
end
is_setindex!_expr(x) = false
is_getindex_expr(expr::Expr) = (expr.head === :ref)
is_getindex_expr(x) = false
is_assignment_expr(expr::Expr) = (expr.head === :(=))
is_assignment_expr(expr) = false

# TODO: Define this in terms of a function mapping
# preserve_graph_function(::typeof(setindex!)) = setindex!_preserve_graph
# preserve_graph_function(::typeof(map_vertex_data)) = map_vertex_data_preserve_graph
# Also allow annotating codeblocks like `@views`.
macro preserve_graph(expr)
    if !is_setindex!_expr(expr)
        error(
            "preserve_graph must be used with setindex! syntax (as @preserve_graph a[i,j,...] = value)",
        )
    end
    @capture(expr, array_[indices__] = value_)
    return :(setindex_preserve_graph!($(esc(array)), $(esc(value)), $(esc.(indices)...)))
end

# Update the graph of the TensorNetwork `tn` to include
# edges that should exist based on the tensor connectivity.
function add_missing_edges!(tn::AbstractGraph)
    foreach(v -> add_missing_edges!(tn, v), vertices(tn))
    return tn
end

# Update the graph of the TensorNetwork `tn` to include
# edges that should be incident to the vertex `v`
# based on the tensor connectivity.
function add_missing_edges!(tn::AbstractGraph, v)
    for v′ in vertices(tn)
        if v ≠ v′
            e = v => v′
            if !isempty(linkinds(tn, e))
                add_edge!(tn, e)
            end
        end
    end
    return tn
end

# Fix the edges of the TensorNetwork `tn` to match
# the tensor connectivity.
function fix_edges!(tn::AbstractGraph)
    foreach(v -> fix_edges!(tn, v), vertices(tn))
    return tn
end
# Fix the edges of the TensorNetwork `tn` to match
# the tensor connectivity at vertex `v`.
function fix_edges!(tn::AbstractGraph, v)
    rem_edges!(tn, incident_edges(tn, v))
    add_missing_edges!(tn, v)
    return tn
end

# Customization point.
using NamedDimsArrays: AbstractNamedUnitRange, namedunitrange, nametype, randname
function trivial_unitrange(type::Type{<:AbstractUnitRange})
    return Base.oneto(one(eltype(type)))
end
function rand_trivial_namedunitrange(
        ::Type{<:AbstractNamedUnitRange{<:Any, R, N}}
    ) where {R, N}
    return namedunitrange(trivial_unitrange(R), randname(N))
end

dag(x) = x

function insert_trivial_link!(tn, e)
    add_edge!(tn, e)
    l = rand_trivial_namedunitrange(eltype(inds(tn[src(e)])))
    x = similar(tn[src(e)], (l,))
    x[1] = 1
    @preserve_graph tn[src(e)] = tn[src(e)] * x
    @preserve_graph tn[dst(e)] = tn[dst(e)] * dag(x)
    return tn
end

function Base.setindex!(tn::AbstractTensorNetwork, value, v)
    @preserve_graph tn[v] = value
    fix_edges!(tn, v)
    return tn
end
# Fix ambiguity error.
function Base.setindex!(graph::AbstractTensorNetwork, value, vertex::OrdinalSuffixedInteger)
    graph[vertices(graph)[vertex]] = value
    return graph
end
Base.setindex!(tn::AbstractTensorNetwork, value, edge::AbstractEdge) = not_implemented()
Base.setindex!(tn::AbstractTensorNetwork, value, edge::Pair) = not_implemented()
# Fix ambiguity error.
function Base.setindex!(
        tn::AbstractTensorNetwork,
        value,
        edge::Pair{<:OrdinalSuffixedInteger, <:OrdinalSuffixedInteger},
    )
    return not_implemented()
end

function Base.show(io::IO, mime::MIME"text/plain", graph::AbstractTensorNetwork)
    println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
    show(io, mime, vertices(graph))
    println(io, "\n")
    println(io, "and $(ne(graph)) edge(s):")
    for e in edges(graph)
        show(io, mime, e)
        println(io)
    end
    println(io)
    println(io, "with vertex data:")
    show(io, mime, axes.(vertex_data(graph)))
    return nothing
end

Base.show(io::IO, graph::AbstractTensorNetwork) = show(io, MIME"text/plain"(), graph)

function Graphs.induced_subgraph(graph::AbstractTensorNetwork, subvertices::AbstractVector{V}) where {V <: Int}
    return tensornetwork_induced_subgraph(graph, subvertices)
end
function Graphs.induced_subgraph(graph::AbstractTensorNetwork, subvertices)
    return tensornetwork_induced_subgraph(graph, subvertices)
end

function tensornetwork_induced_subgraph(graph, subvertices)
    underlying_subgraph, vlist = Graphs.induced_subgraph(underlying_graph(graph), subvertices)
    subgraph = similar_type(graph)(underlying_subgraph)
    for v in vertices(subgraph)
        if isassigned(graph, v)
            set!(vertex_data(subgraph), v, graph[v])
        end
    end
    return subgraph, vlist
end
