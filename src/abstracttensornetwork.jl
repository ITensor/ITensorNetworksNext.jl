using Adapt: Adapt, adapt
using DataGraphs: DataGraphs, AbstractDataGraph, AbstractEdgeDataGraph,
    AbstractVertexDataGraph, EdgeDataGraph, VertexDataGraph, edge_data, set_vertex_data!,
    underlying_graph, underlying_graph_type, vertex_data
using Dictionaries: Dictionaries, Dictionary
using Graphs: Graphs, AbstractEdge, AbstractGraph, add_edge!, add_vertex!, dst, edges,
    edgetype, ne, neighbors, nv, rem_edge!, src, vertices
using ITensorBase: dimnames, inds, name, named, nametype, prime, uniquename, unnamedtype
using LinearAlgebra: LinearAlgebra
using MacroTools: @capture
using NamedGraphs: NamedGraphs, NamedGraph, add_edges!, arrange_edge, decoded_vertex,
    not_implemented, similar_graph, vertextype
using TensorAlgebra: trivialrange

abstract type AbstractITensorNetwork{T, V} <: AbstractVertexDataGraph{T, V} end

# ====================================== Graphs.jl ======================================= #

# Need to be careful about removing edges from tensor networks in case there is a bond
Graphs.rem_edge!(::AbstractITensorNetwork, _edge) = not_implemented()

function Graphs.weights(graph::AbstractITensorNetwork)
    V = vertextype(graph)
    es = Tuple.(edges(graph))
    ws = Dictionary{Tuple{V, V}, Float64}(es, undef)
    for e in edges(graph)
        w = log2(dim(internalinds(graph, e)))
        ws[(src(e), dst(e))] = w
    end
    return ws
end

# Overload if needed
Graphs.is_directed(::Type{<:AbstractITensorNetwork}) = false

# ==================================== NamedGraphs.jl ==================================== #

function NamedGraphs.similar_graph(::AbstractITensorNetwork, VD::Type, vertices)
    return ITensorNetwork{VD}(undef, collect(vertices))
end

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.underlying_graph(tn::AbstractITensorNetwork)
    ug = NamedGraph(vertices(tn))
    add_edges!(ug, edges(tn))
    return ug
end

# ====================================== interface ======================================= #

internalinds(tn::AbstractGraph, edge::Pair) = internalinds(tn, edgetype(tn)(edge))
# Pick the internal indices from the `src` side, identified by name match with `dst`.
# A range-strict intersection (`inds(src) ∩ inds(dst)`) would drop graded internal indices
# whose two endpoints carry dual-related ranges.
function internalinds(tn::AbstractGraph, edge::AbstractEdge)
    ln = internalnames(tn, edge)
    return [i for i in inds(tn[src(edge)]) if name(i) in ln]
end

function internalaxes(tn::AbstractGraph, edge::Pair)
    return internalaxes(tn, edgetype(tn)(edge))
end
function internalaxes(tn::AbstractGraph, edge::AbstractEdge)
    ln = internalnames(tn, edge)
    return [ax for ax in axes(tn[src(edge)]) if name(ax) in ln]
end
function internalnames(tn::AbstractGraph, edge::Pair)
    return internalnames(tn, edgetype(tn)(edge))
end
function internalnames(tn::AbstractGraph, edge::AbstractEdge)
    return dimnames(tn[src(edge)]) ∩ dimnames(tn[dst(edge)])
end

function externalinds(tn::AbstractGraph, v)
    s = inds(tn[v])
    for v′ in neighbors(tn, v)
        s = setdiff(s, inds(tn[v′]))
    end
    return s
end
"""
    InternalIndsGraph

The whole-network internal indices, one entry per edge. Undirected, because the two arrows of
an edge are not independent: `internalinds(tn, e)` returns `src(e)`'s copy of the bond, so the
reverse arrow holds its dual. Storing one arrow and deriving the other through
`reverse_data_direction` keeps that relationship true by construction, where storing both
would let them drift apart.
"""
struct InternalIndsGraph{T, V} <: AbstractEdgeDataGraph{T, V}
    parent::EdgeDataGraph{T, V}
end
Base.parent(g::InternalIndsGraph) = g.parent
Graphs.is_directed(::Type{<:InternalIndsGraph}) = false
DataGraphs.reverse_data_direction(::InternalIndsGraph, is) = conj.(is)

# `AbstractEdgeDataGraph` has no generic support for wrapper types, so each method
# `EdgeDataGraph` defines has to be forwarded to the parent by hand.
function Graphs.edgetype(::Type{<:InternalIndsGraph{T, V}}) where {T, V}
    return edgetype(EdgeDataGraph{T, V})
end
DataGraphs.edge_data_type(::Type{<:InternalIndsGraph{T}}) where {T} = T
function NamedGraphs.similar_graph(g::InternalIndsGraph, T::Type)
    return InternalIndsGraph(similar_graph(parent(g), T))
end
function NamedGraphs.similar_graph(::InternalIndsGraph, T::Type, vertices)
    return InternalIndsGraph(similar_graph(EdgeDataGraph{T}, vertices))
end
for f in (
        :(Graphs.vertices), :(NamedGraphs.encoded_graph), :(DataGraphs.edge_data),
        :(Dictionaries.isinsertable),
    )
    @eval $f(g::InternalIndsGraph) = $f(parent(g))
end
for f in (
        :(Graphs.add_vertex!), :(Graphs.add_edge!), :(Graphs.rem_vertex!),
        :(Graphs.rem_edge!), :(NamedGraphs.encoded_vertex), :(DataGraphs.get_edge_data),
        :(DataGraphs.is_vertex_assigned), :(DataGraphs.is_edge_assigned),
    )
    @eval $f(g::InternalIndsGraph, x) = $f(parent(g), x)
end
# `code` is typed to match the `AbstractDataGraph` method it would otherwise be ambiguous with.
function NamedGraphs.decoded_vertex(g::InternalIndsGraph, code::Integer)
    return decoded_vertex(parent(g), code)
end
for f in (:(DataGraphs.set_edge_data!), :(DataGraphs.insert_edge_data!))
    @eval function $f(g::InternalIndsGraph, x, y)
        $f(parent(g), x, y)
        return g
    end
end

# The whole-network forms, keeping each index associated with the vertex or edge it belongs
# to. External indices hang off a vertex, which has no orientation, so they collect into a
# plain `VertexDataGraph`, which builds only vertices, hence the explicit edges below.
function externalinds(tn::AbstractGraph)
    vs = collect(vertices(tn))
    g = VertexDataGraph(Dictionary(vs, map(v -> externalinds(tn, v), vs)))
    for e in edges(tn)
        add_edge!(g, e)
    end
    return g
end
function internalinds(tn::AbstractGraph)
    # Arranged, because `EdgeDataGraph` files each entry under its edge's arranged direction
    # without transforming it. An entry keyed by the reverse arrow would land in the
    # arranged slot still holding the reverse arrow's indices, and reading it back would
    # then hand out the dual of what `internalinds(tn, edge)` returns.
    es = map(e -> arrange_edge(tn, e), edges(tn))
    return InternalIndsGraph(
        EdgeDataGraph(Dictionary(es, map(e -> internalinds(tn, e), es)))
    )
end

function externalaxes(tn::AbstractGraph, v)
    s = axes(tn[v])
    for v′ in neighbors(tn, v)
        s = setdiff(s, axes(tn[v′]))
    end
    return s
end
function externalnames(tn::AbstractGraph, v)
    s = dimnames(tn[v])
    for v′ in neighbors(tn, v)
        s = setdiff(s, dimnames(tn[v′]))
    end
    return s
end

# Return the vertices associated with a dim name
function dimnamevertices(tn::AbstractGraph, name)
    vs = vertextype(tn)[]

    for v in vertices(tn)
        if name ∈ dimnames(tn[v])
            push!(vs, v)
        end
    end

    return vs
end

function has_dimname(tn::AbstractGraph, name)
    for v in vertices(tn)
        if name ∈ dimnames(tn[v])
            return true
        end
    end
    return false
end

has_ind(tn::AbstractGraph, ind) = has_dimname(tn, name(ind))

function insertinternalind!(tn::AbstractGraph, e)
    T = eltype(inds(tn[src(e)]))

    internalind = named(trivialrange(unnamedtype(T)), uniquename(nametype(T)))

    x = similar(tn[src(e)], (internalind,))
    fill!(x, true)

    tn[src(e)] *= x
    tn[dst(e)] *= conj(x)

    return tn
end
