using ..ITensorNetworksNext: TensorNetwork
using FunctionImplementations: zero!
using Graphs: AbstractGraph
using ITensorBase: NamedUnitRange, denamed, name, nameddims
using NamedGraphs.GraphsExtensions: incident_edges

diaglength(a::AbstractArray) = minimum(size(a))
function diagstride(a::AbstractArray)
    s = 1
    p = 1
    for i in 1:(ndims(a) - 1)
        p *= size(a, i)
        s += p
    end
    return s
end
function diagindices(a::AbstractArray)
    maxdiag = LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength(a)), ndims(a)))]
    return 1:diagstride(a):maxdiag
end
diagview(a::AbstractArray) = @view a[diagindices(a)]

function diagonaltensor(diag::AbstractVector, ax::Tuple{Vararg{AbstractUnitRange}})
    a = similar(diag, ax)
    zero!(a)
    diagview(a) .= diag
    return a
end
function diagonaltensor(
        diag::AbstractVector,
        is::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return nameddims(diagonaltensor(diag, denamed.(is)), name.(is))
end

delta(elt::Type, is) = diagonaltensor(ones(elt, minimum(length ∘ denamed, is)), is)

"""
    delta_network(f, elt::Type = Float64, g::AbstractGraph)

Construct a TensorNetwork on the graph `g` with element type `elt` that has delta tensors
on each vertex. Link dimensions are defined using the function `f(e)` that should take an
edge `e` as an input and should output the link index on that edge.
"""
function delta_network(f, elt::Type, g::AbstractGraph)
    return tn = TensorNetwork(g) do v
        is = Tuple(f.(incident_edges(g, v)))
        return delta(elt, is)
    end
end
function delta_network(f, g::AbstractGraph)
    return delta_network(f, Float64, g)
end
