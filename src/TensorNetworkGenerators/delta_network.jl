using ..ITensorNetworksNext: TensorNetwork
using DiagonalArrays: δ
using Graphs: AbstractGraph
using NamedGraphs.GraphsExtensions: incident_edges

"""
    delta_network(f, elt::Type = Float64, g::AbstractGraph)

Construct a TensorNetwork on the graph `g` with element type `elt` that has delta tensors
on each vertex. Link dimensions are defined using the function `f(e)` that should take an
edge `e` as an input and should output the link index on that edge.
"""
function delta_network(f, elt::Type, g::AbstractGraph)
    return tn = TensorNetwork(g) do v
        is = Tuple(f.(incident_edges(g, v)))
        return δ(elt, is)
    end
end
function delta_network(f, g::AbstractGraph)
    return delta_network(f, Float64, g)
end
