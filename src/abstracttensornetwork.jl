using Adapt: Adapt, adapt
using DataGraphs: DataGraphs, AbstractDataGraph, AbstractVertexDataGraph, edge_data,
    set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data
using Dictionaries: Dictionary
using Graphs: Graphs, AbstractEdge, AbstractGraph, add_edge!, add_vertex!, dst, edges,
    edgetype, ne, neighbors, nv, rem_edge!, src, vertices
using ITensorBase: denamedtype, dimnames, inds, name, named, nametype, prime, uniquename
using LinearAlgebra: LinearAlgebra
using MacroTools: @capture
using NamedGraphs.GraphsExtensions: directed_graph, incident_edges, rem_edges!, vertextype
using NamedGraphs.OrdinalIndexing: OrdinalSuffixedInteger
using NamedGraphs: NamedGraphs, NamedGraph, not_implemented, similar_graph
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
        w = log2(dim(linkinds(graph, e)))
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

linkinds(tn::AbstractGraph, edge::Pair) = linkinds(tn, edgetype(tn)(edge))
# Pick the link indices from the `src` side, identified by name match with `dst`.
# A range-strict intersection (`inds(src) ∩ inds(dst)`) would drop graded links
# whose two endpoints carry dual-related ranges.
function linkinds(tn::AbstractGraph, edge::AbstractEdge)
    ln = linknames(tn, edge)
    return [i for i in inds(tn[src(edge)]) if name(i) in ln]
end

function linkaxes(tn::AbstractGraph, edge::Pair)
    return linkaxes(tn, edgetype(tn)(edge))
end
function linkaxes(tn::AbstractGraph, edge::AbstractEdge)
    ln = linknames(tn, edge)
    return [ax for ax in axes(tn[src(edge)]) if name(ax) in ln]
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

function has_dimname(tn::AbstractGraph, name)
    for v in vertices(tn)
        if name ∈ dimnames(tn[v])
            return true
        end
    end
    return false
end

has_ind(tn::AbstractGraph, ind) = has_dimname(tn, name(ind))

function insertlink!(tn::AbstractGraph, e)
    T = eltype(inds(tn[src(e)]))

    linkind = named(trivialrange(denamedtype(T)), uniquename(nametype(T)))

    x = similar(tn[src(e)], (linkind,))
    fill!(x, true)

    tn[src(e)] *= x
    tn[dst(e)] *= conj(x)

    return tn
end
