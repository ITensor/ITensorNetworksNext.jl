using ..ITensorNetworksNext
using Graphs: degree, dst, edges, src
using ITensorBase: name, nameddims, uniquename
using LinearAlgebra: Diagonal, eigen
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

Construct a ITensorNetwork on the graph `g` with inverse temperature `β` that has Ising
partition function tensors on each vertex. Link dimensions are defined using the function
`f(e)` that should take an edge `e` as an input and should output the link index on that
edge.
"""
function ising_network(
        f, β::Number, g::AbstractGraph; J::Number = one(β), h::Number = zero(β),
        sz_vertices = vertextype(g)[]
    )
    elt = typeof(β)
    l̃ = Dict(e => uniquename(f(e)) for e in edges(g))

    fp(e) = get(() -> l̃[reverse(e)], l̃, e)
    tn = delta_network(fp, elt, g)
    for v in sz_vertices
        tn[v] = diagonaltensor(elt[1, -1], axes(tn[v]))
    end

    for e in edges(g)
        v1 = src(e)
        v2 = dst(e)
        deg1 = degree(tn, v1)

        deg2 = degree(tn, v2)
        m = sqrt_ising_bond(β; J, h, deg1, deg2)
        # Split the Ising bond as √b on each endpoint, contracting the delta-network bond
        # name `fp(e)` and renaming the shared bond to the requested name `f(e)`.
        b = nameddims(m, (name(fp(e)), name(f(e))))
        tn[v1] = b * tn[v1]
        tn[v2] = b * tn[v2]
    end
    return tn
end
