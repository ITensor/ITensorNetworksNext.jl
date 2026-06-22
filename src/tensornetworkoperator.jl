using DataGraphs: DataGraphs, underlying_graph
using NamedDimsArrays: NamedDimsArrays as NDA
using NamedGraphs.GraphsExtensions: vertextype

# A tensor-network operator: an operator `TensorNetwork` together with the map between its
# bra-side and ket-side physical names. The tensor-network analogue of a `NamedDimsArrays`
# operator, with `state` the underlying operator network (mirroring `NamedDimsArrays.state`
# on a `NamedDimsOperator`) and the tensor-network interface forwarded to it.
struct TensorNetworkOperator{V, VD, State, SiteMap} <: AbstractTensorNetwork{V, VD}
    state::State
    site_index_map::SiteMap
end

NDA.state(o::TensorNetworkOperator) = o.state
site_index_map(o::TensorNetworkOperator) = o.site_index_map

DataGraphs.underlying_graph(o::TensorNetworkOperator) = underlying_graph(NDA.state(o))
function DataGraphs.is_vertex_assigned(o::TensorNetworkOperator, v)
    return DataGraphs.is_vertex_assigned(NDA.state(o), v)
end
function DataGraphs.is_edge_assigned(o::TensorNetworkOperator, e)
    return DataGraphs.is_edge_assigned(NDA.state(o), e)
end
function DataGraphs.get_vertex_data(o::TensorNetworkOperator, v)
    return DataGraphs.get_vertex_data(NDA.state(o), v)
end

function Base.copy(o::TensorNetworkOperator)
    return TensorNetworkOperator(copy(NDA.state(o)), site_index_map(o))
end

function TensorNetworkOperator(state, site_index_map)
    return TensorNetworkOperator{
        vertextype(state), eltype(state), typeof(state), typeof(site_index_map),
    }(
        state, site_index_map
    )
end
