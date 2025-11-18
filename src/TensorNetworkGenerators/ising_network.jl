using DiagonalArrays: DiagonalArray
using Graphs: degree, dst, edges, src
using ..ITensorNetworksNext: @preserve_graph
using LinearAlgebra: Diagonal, eigen
using NamedDimsArrays: apply, dename, inds, operator, randname
using NamedGraphs.GraphsExtensions: vertextype

function sqrt_ising_bond(β; J = one(β), h = zero(β), deg1::Integer, deg2::Integer)
    h1 = h / deg1
    h2 = h / deg2
    m = [
        exp(β * (J + h1 + h2)) exp(β * (-J + h1 - h2));
        exp(β * (-J - h1 + h2)) exp(β * (J - h1 - h2));
    ]
    d, v = eigen(m)
    return v * √(Diagonal(d)) * inv(v)
end

"""
    ising_network(f, β::Number, g::AbstractGraph)

Construct a TensorNetwork on the graph `g` with inverse temperature `β` that has Ising
partition function tensors on each vertex. Link dimensions are defined using the function
`f(e)` that should take an edge `e` as an input and should output the link index on that
edge.
"""
function ising_network(
        f, β::Number, g::AbstractGraph; J::Number = one(β), h::Number = zero(β),
        sz_vertices = vertextype(g)[]
    )
    elt = typeof(β)
    l̃ = Dict(e => randname(f(e)) for e in edges(g))
    f̃(e) = get(() -> l̃[reverse(e)], l̃, e)
    tn = delta_network(f̃, elt, g)
    for v in sz_vertices
        a = DiagonalArray(elt[1, -1], dename.(inds(tn[v])))
        tn[v] = a[inds(tn[v])...]
    end
    for e in edges(tn)
        v1 = src(e)
        v2 = dst(e)
        deg1 = degree(tn, v1)
        deg2 = degree(tn, v2)
        m = sqrt_ising_bond(β; J, h, deg1, deg2)
        t = operator(m, (f̃(e),), (f(e),))
        @preserve_graph tn[v1] = apply(t, tn[v1])
        @preserve_graph tn[v2] = apply(t, tn[v2])
    end
    return tn
end
