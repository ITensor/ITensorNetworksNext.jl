using .LazyNamedDimsArrays: Mul, lazy, materialize, optimize_evaluation_order
using Combinatorics: combinations
using DataGraphs.DataGraphsPartitionedGraphsExt
using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph, edge_data, get_vertices_data,
    vertex_data, vertex_data_type
using Dictionaries:
    Dictionaries, AbstractDictionary, Dictionary, Indices, dictionary, set!, unset!
using Graphs:
    AbstractSimpleGraph, SimpleGraph, edges, has_edge, rem_edge!, rem_vertex!, vertices
using NamedDimsArrays:
    NamedDimsArrays, AbstractNamedDimsArray, denamedtype, dim, dimnames, name, nametype
using NamedGraphs.GraphsExtensions:
    GraphsExtensions, arrange_edge, arranged_edges, vertextype
using NamedGraphs.OrderedDictionaries:
    OrderedDictionary, OrderedIndices, index_positions, ordered_indices
using NamedGraphs.PartitionedGraphs: AbstractPartitionedGraph, PartitionedGraphs,
    QuotientVertex, QuotientVertexVertices, QuotientVertices, departition,
    partitioned_vertices, partitionedgraph, quotient_graph, quotient_graph_type,
    quotientvertices
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, PositionGraphView, Vertices,
    parent_graph_indices, vertextype
using SplitApplyCombine: mapview

struct TensorNetwork{T, V, I} <: AbstractTensorNetwork{T, V}
    tensors::Dictionary{V, T}
    index_locations::Dictionary{I, Set{V}}
    underlying_graph::NamedGraph{V}
    function TensorNetwork{T, V, I}(::UndefInitializer, vertices) where {T, V, I}
        tensors = Dictionary{V, T}()
        index_locations = Dictionary{I, Set{V}}()
        underlying_graph = NamedGraph(vertices)
        return new{T, V, I}(tensors, index_locations, underlying_graph)
    end
end

function TensorNetwork{T}(undef::UndefInitializer, vertices) where {T}
    return TensorNetwork{T, eltype(vertices)}(undef, vertices)
end

function TensorNetwork{T, V}(undef::UndefInitializer, vertices) where {T, V}
    return TensorNetwork{T, V, nametype(T)}(undef, vertices)
end

TensorNetwork(tensors) = TensorNetwork{valtype(tensors)}(tensors)
TensorNetwork{T}(tensors) where {T} = TensorNetwork{T, keytype(tensors)}(tensors)
function TensorNetwork{T, V}(tensors) where {T, V}
    I = nametype(T)
    tn = TensorNetwork{T, V, I}(undef, keys(tensors))
    copyto!(tn, tensors)
    return tn
end

NamedDimsArrays.nametype(::Type{<:TensorNetwork{T, V, I}}) where {T, V, I} = I

Graphs.vertices(tn::TensorNetwork) = vertices(tn.underlying_graph)

function NamedGraphs.vertex_positions(graph::TensorNetwork)
    return index_positions(vertices(graph))
end
function NamedGraphs.ordered_vertices(graph::TensorNetwork)
    return ordered_indices(vertices(graph))
end

NamedGraphs.position_graph(graph::TensorNetwork) = position_graph(graph.underlying_graph)

function Base.copy(tn::TensorNetwork{T}) where {T}
    tn_dst = TensorNetwork{T}(undef, vertices(tn))
    copyto!(tn_dst, tn)
    return tn_dst
end

function Graphs.rem_vertex!(tn::TensorNetwork, vertex)
    tensor = tn.tensors[vertex]

    for ind in dimnames(tensor)

        # If `ind` is associated with an edge, remove the edge.
        delete_ind_edge!(tn, ind)

        # Delete the vertex from that `ind`s vertex list
        # (this index may still be one incident to one other vertex)
        vertex_list = tn.index_locations[ind]
        delete!(vertex_list, vertex)

        # If that index is now no longer associated with any vertices, it was dangling,
        # and that index should be deleted from the keys of reverse index mapping
        isempty(vertex_list) && delete!(tn.index_locations, ind)
    end

    rem_vertex!(tn.underlying_graph, vertex)
    delete!(tn.tensors, vertex)

    return tn
end

# Internal (unsafe)
function delete_ind_edge!(tn, ind)
    vertex_list = tn.index_locations[ind]

    if length(vertex_list) == 2
        src, dst = vertex_list
        rem_edge!(tn.underlying_graph, src => dst)
    end

    return tn
end

# Internal (unsafe)
function delete_ind_vertex!(tn, ind, vertex)
    vertex_list = tn.index_locations[ind]

    delete!(vertex_list, vertex)
    isempty(vertex_list) && delete!(tn.index_locations, ind)

    return tn
end

tensornetwork(f, vertices) = TensorNetwork(Dict(v => f(v) for v in vertices))

Graphs.is_directed(::Type{<:TensorNetwork}) = false

# ====================================== DataGraphs ====================================== #

DataGraphs.is_vertex_assigned(tn::TensorNetwork, vertex) = isassigned(tn.tensors, vertex)
DataGraphs.is_edge_assigned(::TensorNetwork, _edge) = false

DataGraphs.get_vertex_data(tn::TensorNetwork, v) = tn.tensors[v]

function DataGraphs.insert_vertex_data!(tn::TensorNetwork, vertex, tensor)
    add_vertex!(tn.underlying_graph, vertex)
    set!_tensornetwork(tn, vertex, tensor)
    return tn
end

function DataGraphs.set_vertex_data!(tn::TensorNetwork, tensor, vertex)
    set!_tensornetwork(tn, vertex, tensor)
    return tn
end

# "upsert"
function set!_tensornetwork(tn::TensorNetwork, vertex, tensor)
    newinds = dimnames(tensor)

    oldinds = get(mapview(dimnames, tn.tensors), vertex, Set())

    # Only have to deal with the indices that aren't shared.
    for ind in symdiff(oldinds, newinds)
        if ind in oldinds
            delete_ind_edge!(tn, ind)
            delete_ind_vertex!(tn, ind, vertex)
            continue
        end

        # Now `ind` must be a new index that's not in `oldinds`

        vertex_list = get!(tn.index_locations, ind, Set())
        if length(vertex_list) > 1
            throw(
                ArgumentError(
                    "index $ind can appear in at most one existing tensor, got $(length(vertex_list))."
                )
            )
        end
        push!(vertex_list, vertex)

        # Add an edge if the index is now shared between two vertices.
        if length(vertex_list) == 2
            src, dst = vertex_list
            add_edge!(tn.underlying_graph, src, dst)
        end
    end

    set!(tn.tensors, vertex, tensor)

    return tn
end

Dictionaries.isinsertable(::TensorNetwork) = true

function DataGraphs.underlying_graph_type(type::Type{<:TensorNetwork{T, V}}) where {T, V}
    return fieldtype(type, :underlying_graph)
end

function Graphs.rem_edge!(::TensorNetwork, _edge)
    return throw(
        ErrorException("removing edges from the `TensorNetwork` type is not supported.")
    )
end

function Graphs.add_edge!(::TensorNetwork, _edge)
    return throw(
        ErrorException("Adding edges to the `TensorNetwork` type is not supported.")
    )
end

# PERF: fast lookup compared to `AbstractTensorNetwork` fallback.
indsites(tn::TensorNetwork, ind) = tn.index_locations[name(ind)]

# PERF: fast lookup compared to `AbstractTensorNetwork` fallback.
has_ind(tn::TensorNetwork, ind) = haskey(tn.index_locations, name(ind))

function NamedGraphs.similar_graph(
        T::Type{<:TensorNetwork},
        vertices = vertextype(T)[]
    )
    return T(undef, vertices)
end
function NamedGraphs.similar_graph(::TensorNetwork, VD::Type, vertices)
    return TensorNetwork{VD}(undef, collect(vertices))
end

function NamedGraphs.convert_vertextype(V::Type, tn_src::TensorNetwork{T}) where {T}
    tn_dst = TensorNetwork{eltype(tn_src), V}(undef, vertices(tn_src))
    copyto!(tn_dst, tn_src)
    return tn_dst
end

function NamedGraphs.induced_subgraph_from_vertices(tn::TensorNetwork, subvertices)
    subgraph = similar_graph(tn, subvertices)
    copyto!(subgraph, tn, subvertices)
    return subgraph, subvertices
end
