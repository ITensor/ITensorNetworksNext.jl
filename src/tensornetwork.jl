using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph
using Dictionaries: Indices, dictionary
## using ITensors: ITensors, ITensor, op, state
## using .ITensorsExtensions: trivial_space
using NamedDimsArrays: AbstractNamedDimsArray
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype
using NamedGraphs.GraphsExtensions: vertextype

function _TensorNetwork end

struct TensorNetwork{V,VD,DG<:AbstractDataGraph{V,VD}} <: AbstractTensorNetwork{V,VD}
  data_graph::DG
  global @inline function _TensorNetwork(
    data_graph::DG
  ) where {DG<:AbstractDataGraph{V,VD}} where {V,VD}
    return new{V,VD,DG}(data_graph)
  end
end

#
# Data access
#

data_graph(tn::TensorNetwork) = getfield(tn, :data_graph)
data_graph_type(TN::Type{<:TensorNetwork}) = fieldtype(TN, :data_graph)

function DataGraphs.underlying_graph_type(TN::Type{<:TensorNetwork})
  return fieldtype(data_graph_type(TN), :underlying_graph)
end

# Versions taking vertex types.
function TensorNetwork{V}() where {V}
  # TODO: Is there a better way to write this?
  # Try using `convert_vertextype`.
  new_data_graph_type = data_graph_type(TensorNetwork{V})
  new_underlying_graph_type = underlying_graph_type(new_data_graph_type)
  return _TensorNetwork(new_data_graph_type(new_underlying_graph_type()))
end
function TensorNetwork{V}(tn::TensorNetwork) where {V}
  # TODO: Is there a better way to write this?
  # Try using `convert_vertextype`.
  return _TensorNetwork(DataGraph{V}(data_graph(tn)))
end
function TensorNetwork{V}(g::NamedGraph) where {V}
  # TODO: Is there a better way to write this?
  # Try using `convert_vertextype`.
  return TensorNetwork(NamedGraph{V}(g))
end

TensorNetwork() = TensorNetwork{Any}()

# Conversion
# TODO: Copy or not?
TensorNetwork(tn::TensorNetwork) = copy(tn)

NamedGraphs.convert_vertextype(::Type{V}, tn::TensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::TensorNetwork) = TensorNetwork{V}(tn)

Base.copy(tn::TensorNetwork) = _TensorNetwork(copy(data_graph(tn)))

#
# Construction from collections of tensors
#

function tensors_to_tensornetwork(ts)
  g = NamedGraph(collect(eachindex(ts)))
  tn = TensorNetwork(g)
  for v in vertices(g)
    tn[v] = ts[v]
  end
  return tn
end
function TensorNetwork(ts::AbstractVector{<:AbstractNamedDimsArray})
  return tensors_to_tensornetwork(ts)
end
function TensorNetwork(ts::AbstractDictionary{<:Any,<:AbstractNamedDimsArray})
  return tensors_to_tensornetwork(ts)
end
function TensorNetwork(ts::AbstractDict{<:Any,<:AbstractNamedDimsArray})
  return tensors_to_tensornetwork(ts)
end
function TensorNetwork(vs::AbstractVector, ts::AbstractVector{<:AbstractNamedDimsArray})
  return tensors_to_tensornetwork(Dictionary(vs, ts))
end
function TensorNetwork(ts::AbstractVector{<:Pair{<:Any,<:AbstractNamedDimsArray}})
  return tensors_to_tensornetwork(dictionary(ts))
end
# TODO: Decide what this should do, maybe it should factorize?
function TensorNetwork(t::AbstractNamedDimsArray)
  return tensors_to_tensornetwork([t])
end

#
# Construction from underyling named graph
#

function TensorNetwork(
  eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...
)
  return TensorNetwork(eltype, undef, IndsNetwork(graph; kwargs...))
end

function TensorNetwork(f, graph::AbstractNamedGraph; kwargs...)
  return TensorNetwork(f, IndsNetwork(graph; kwargs...))
end

function TensorNetwork(graph::AbstractNamedGraph; kwargs...)
  return TensorNetwork(IndsNetwork(graph; kwargs...))
end

#
# Construction from underyling simple graph
#

function TensorNetwork(
  eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kwargs...
)
  return TensorNetwork(eltype, undef, IndsNetwork(graph; kwargs...))
end

function TensorNetwork(f, graph::AbstractSimpleGraph; kwargs...)
  return TensorNetwork(f, IndsNetwork(graph); kwargs...)
end

function TensorNetwork(graph::AbstractSimpleGraph; kwargs...)
  return TensorNetwork(IndsNetwork(graph); kwargs...)
end

#
# Construction from IndsNetwork
#

function TensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  return TensorNetwork(is; kwargs...) do v
    return (inds...) -> nameddimsarray(eltype, undef, inds...)
  end
end

function TensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  return TensorNetwork(is; kwargs...) do v
    return (inds...) -> nameddimsarray(eltype, inds...)
  end
end

function TensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  return TensorNetwork(is; kwargs...) do v
    return (inds...) -> nameddimsarray(undef, inds...)
  end
end

function TensorNetwork(is::IndsNetwork; kwargs...)
  return TensorNetwork(is; kwargs...) do v
    return (inds...) -> nameddimsarray(inds...)
  end
end

# TODO: Handle `eltype` and `undef` through `generic_state`.
# `inds` are stored in a `NamedTuple`
function generic_state(f, inds::NamedTuple)
  return generic_state(f, reduce(vcat, inds.linkinds; init=inds.siteinds))
end

function generic_state(f, inds::Vector)
  return f(inds)
end
function generic_state(a::AbstractArray, inds::Vector)
  return itensor(a, inds)
end
## function generic_state(x::Op, inds::NamedTuple)
##   # TODO: Figure out what to do if there is more than one site.
##   if !isempty(inds.siteinds)
##     @assert length(inds.siteinds) == 2
##     i = inds.siteinds[findfirst(i -> plev(i) == 0, inds.siteinds)]
##     @assert i' ∈ inds.siteinds
##     site_tensors = [op(x.which_op, i)]
##   else
##     site_tensors = []
##   end
##   link_tensors = [[onehot(i => 1) for i in inds.linkinds[e]] for e in keys(inds.linkinds)]
##   return contract(reduce(vcat, link_tensors; init=site_tensors))
## end
function generic_state(s::AbstractString, inds::NamedTuple)
  # TODO: Figure out what to do if there is more than one site.
  site_tensors = [state(s, only(inds.siteinds))]
  link_tensors = [[onehot(i => 1) for i in inds.linkinds[e]] for e in keys(inds.linkinds)]
  return contract(reduce(vcat, link_tensors; init=site_tensors))
end

# TODO: This is similar to `ModelHamiltonians.to_callable`,
# try merging the two.
to_callable(value::Type) = value
to_callable(value::Function) = value
function to_callable(value::AbstractDict)
  return Base.Fix1(getindex, value) ∘ keytype(value)
end
function to_callable(value::AbstractDictionary)
  return Base.Fix1(getindex, value) ∘ keytype(value)
end
function to_callable(value::AbstractArray{<:Any,N}) where {N}
  return Base.Fix1(getindex, value) ∘ CartesianIndex
end
to_callable(value) = Returns(value)

function TensorNetwork(value, is::IndsNetwork; kwargs...)
  return TensorNetwork(to_callable(value), is; kwargs...)
end

function TensorNetwork(
  elt::Type, f, is::IndsNetwork; link_space=trivial_space(is), kwargs...
)
  tn = TensorNetwork(f, is; kwargs...)
  for v in vertices(tn)
    tn[v] = elt.(tn[v])
  end
  return tn
end

function TensorNetwork(
  itensor_constructor::Function,
  is::IndsNetwork;
  link_space=indtype(is)(Base.OneTo(1)),
  kwargs...,
)
  is = insert_linkinds(is; link_space)
  tn = TensorNetwork{vertextype(is)}()
  for v in vertices(is)
    add_vertex!(tn, v)
  end
  for e in edges(is)
    add_edge!(tn, e)
  end
  for v in vertices(tn)
    # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
    siteinds = get(is, v, Index[])
    edges = [edgetype(is)(v, nv) for nv in neighbors(is, v)]
    linkinds = map(e -> is[e], Indices(edges))
    tensor_v = generic_state(itensor_constructor(v), (; siteinds, linkinds))
    setindex_preserve_graph!(tn, tensor_v, v)
  end
  return tn
end

TensorNetwork(itns::Vector{<:TensorNetwork}) = reduce(⊗, itns)

# TODO: Use `vertex_data` here?
function eachtensor(ψ::TensorNetwork)
  # This type declaration is needed to narrow
  # the element type of the resulting `Dictionary`,
  # raise and issue with `Dictionaries.jl`.
  return map(v -> ψ[v], vertices(ψ))
end
