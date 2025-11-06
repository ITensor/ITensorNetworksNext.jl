using DiagonalArrays: DiagonalArray
using Graphs: degree, dst, edges, src
using LinearAlgebra: Diagonal, eigen
using NamedDimsArrays: apply, dename, inds, operator, randname

function sqrt_ising_bond(β; h1 = zero(typeof(β)), h2 = zero(typeof(β)))
    f11 = exp(β * (1 + h1 + h2))
    f12 = exp(β * (-1 + h1 - h2))
    f21 = exp(β * (-1 - h1 + h2))
    f22 = exp(β * (1 - h1 - h2))
    m² = eltype(β)[f11 f12; f21 f22]
    d², v = eigen(m²)
    d = sqrt.(d²)
    return v * Diagonal(d) * inv(v)
end

"""
    ising_network(f, β::Number, g::AbstractGraph)

Construct a TensorNetwork on the graph `g` with inverse temperature `β` that has Ising
partition function tensors on each vertex. Link dimensions are defined using the function
`f(e)` that should take an edge `e` as an input and should output the link index on that
edge.
"""
function ising_network(
        f, β::Number, g::AbstractGraph; h::Number = zero(eltype(β)), sz_vertices = []
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
        m = sqrt_ising_bond(β; h1 = h / deg1, h2 = h / deg2)
        t = operator(m, (f̃(e),), (f(e),))
        tn[v1] = apply(t, tn[v1])
        tn[v2] = apply(t, tn[v2])
    end
    return tn
end
