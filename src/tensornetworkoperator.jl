using DataGraphs: DataGraphs, underlying_graph
using ITensorBase: ITensorBase as ITB
using NamedGraphs.GraphsExtensions: vertextype

# A tensor-network operator: an operator `ITensorNetwork` together with the map between its
# bra-side and ket-side physical names. The tensor-network analogue of an ITensor operator,
# with `state` the underlying operator network (mirroring `ITensorBase.state` on an
# `ITensorOperator`) and the tensor-network interface forwarded to it.
struct TensorNetworkOperator{V, VD, State, SiteMap} <: AbstractITensorNetwork{V, VD}
    state::State
    site_index_map::SiteMap
end

ITB.state(o::TensorNetworkOperator) = o.state
site_index_map(o::TensorNetworkOperator) = o.site_index_map

DataGraphs.underlying_graph(o::TensorNetworkOperator) = underlying_graph(ITB.state(o))
function DataGraphs.is_vertex_assigned(o::TensorNetworkOperator, v)
    return DataGraphs.is_vertex_assigned(ITB.state(o), v)
end
function DataGraphs.is_edge_assigned(o::TensorNetworkOperator, e)
    return DataGraphs.is_edge_assigned(ITB.state(o), e)
end
function DataGraphs.get_vertex_data(o::TensorNetworkOperator, v)
    return DataGraphs.get_vertex_data(ITB.state(o), v)
end

function Base.copy(o::TensorNetworkOperator)
    return TensorNetworkOperator(copy(ITB.state(o)), site_index_map(o))
end

function TensorNetworkOperator(state, site_index_map)
    return TensorNetworkOperator{
        vertextype(state), eltype(state), typeof(state), typeof(site_index_map),
    }(
        state, site_index_map
    )
end
