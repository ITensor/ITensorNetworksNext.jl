using Adapt: Adapt, adapt
using BackendSelection: @Algorithm_str, Algorithm
using DataGraphs: DataGraphs, AbstractDataGraph, AbstractVertexDataGraph, edge_data,
    set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data
using Dictionaries: Dictionary
using Graphs: Graphs, AbstractEdge, AbstractGraph, add_edge!, add_vertex!, dst, edges,
    edgetype, ne, neighbors, nv, rem_edge!, src, vertices
using LinearAlgebra: LinearAlgebra
using MacroTools: @capture
using NamedDimsArrays:
    AbstractNamedUnitRange, dimnames, inds, namedunitrange, nametype, randname
using NamedGraphs.GraphsExtensions: directed_graph, incident_edges, rem_edges!, vertextype
using NamedGraphs.OrdinalIndexing: OrdinalSuffixedInteger
using NamedGraphs: NamedGraphs, NamedGraph, not_implemented, similar_graph

abstract type AbstractTensorNetwork{T, V} <: AbstractVertexDataGraph{V, T} end

# ====================================== Graphs.jl ======================================= #

# Need to be careful about removing edges from tensor networks in case there is a bond
Graphs.rem_edge!(::AbstractTensorNetwork, _edge) = not_implemented()

function Graphs.weights(graph::AbstractTensorNetwork)
    V = vertextype(graph)
    es = Tuple.(edges(graph))
    ws = Dictionary{Tuple{V, V}, Float64}(es, undef)
    for e in edges(graph)
        w = log2(dim(linkinds(graph, e)))
        ws[(src(e), dst(e))] = w
    end
    return ws
end

# Overload if needed
Graphs.is_directed(::Type{<:AbstractTensorNetwork}) = false

# Ambiguity stemming from `Graphs.jl`
Graphs.inneighbors(tn::AbstractTensorNetwork, v::Integer) = inneighbors_tensornetwork(tn, v)
Graphs.inneighbors(tn::AbstractTensorNetwork, v) = inneighbors_tensornetwork(tn, v)

function inneighbors_tensornetwork(tn::AbstractGraph, v)
    indices = inds(tn[v])
    in_neighbors = Set{eltype(vertices(tn))}()
    for ind in indices
        vertex_list = indsites(tn, ind)
        union!(in_neighbors, vertex_list)
    end
    return collect(delete!(in_neighbors, v))
end

# Ambiguity stemming from `Graphs.jl`
Graphs.outneighbors(g::AbstractTensorNetwork, v::Integer) = Graphs.inneighbors(g, v)
Graphs.outneighbors(g::AbstractTensorNetwork, v) = Graphs.inneighbors(g, v)

# ==================================== NamedGraphs.jl ==================================== #

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.underlying_graph(tn::AbstractTensorNetwork)
    ug = NamedGraph(vertices(tn))
    add_edges!(ug, edges(tn))
    return ug
end

# ====================================== interface ======================================= #

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
function siteaxes(tn::AbstractGraph, v)
    s = axes(tn[v])
    for v′ in neighbors(tn, v)
        s = setdiff(s, axes(tn[v′]))
    end
    return s
end
function sitenames(tn::AbstractGraph, v)
    s = dimnames(tn[v])
    for v′ in neighbors(tn, v)
        s = setdiff(s, dimnames(tn[v′]))
    end
    return s
end

# Return the vertices associated with an index.
function indsites(tn::AbstractGraph, ind)
    sites = vertextype(tn)[]

    for v in vertices(tn)
        if ind ∈ inds(tn[v])
            push!(sites, v)
        end
    end

    return sites
end

function has_ind(tn::AbstractGraph, ind)
    for v in vertices(tn)
        if ind ∈ inds(tn[v])
            return true
        end
    end
    return false
end

# WARN: this may be ill-defined for fermions
# TODO: Delete (or replace with factorization method)
function add_link!(tn::AbstractTensorNetwork, edge)
    ind = rand_trivial_namedunitrange(eltype(inds(tn[src(edge)])))
    add_link!(tn, edge, ind)
    return tn
end
function add_link!(tn::AbstractTensorNetwork, edge, ind)
    has_ind(tn, ind) && throw(ArgumentError("index $ind already exists"))

    x = similar(tn[src(edge)], (ind,))
    x .= false
    x[1] = true

    new_src = tn[src(edge)] * x
    new_dst = tn[dst(edge)] * x

    tn[src(edge)] = new_src
    tn[dst(edge)] = new_dst

    return tn
end

function trivial_unitrange(type::Type{<:AbstractUnitRange})
    return Base.oneto(one(eltype(type)))
end

function rand_trivial_namedunitrange(
        ::Type{<:AbstractNamedUnitRange{<:Any, R, N}}
    ) where {R, N}
    return namedunitrange(trivial_unitrange(R), randname(N))
end
