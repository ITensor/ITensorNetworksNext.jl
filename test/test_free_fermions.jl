using GradedArrays: gradedrange
using Graphs: dst, edges, src, vertices
using ITensorBase: Index, apply, name, operator, prime
using ITensorNetworksNext:
    NormNetwork, apply_operators, insertlink!, message_environment, tensornetwork
using LinearAlgebra: I
using NamedGraphs: named_comb_tree, named_cycle_graph
using TensorAlgebra: TensorAlgebra as TA
using TensorKitSectors: FermionParity, Z2Irrep
using Test: @test, @testset

# Free fermions hopping on a graph, compared against the single-particle correlation matrix.
# Fermion signs change the densities on a graph with a branching vertex or a loop, so both
# geometries are covered. The circuit is not truncated, so the comparison is exact.

# Two-site matrices in the basis `|n₁ n₂⟩` with the first site fastest.
basis(n1, n2) = 1 + n1 + 2n2
const hopping = let h = zeros(4, 4)
    h[basis(1, 0), basis(0, 1)] = h[basis(0, 1), basis(1, 0)] = 1
    h
end
const pair_creation = let p = zeros(4, 4)
    p[basis(1, 1), basis(0, 0)] = 1
    p
end
# The current `i(c₁† c₂ - c₂† c₁)`. The symmetric hopping `c₁† c₂ + c₂† c₁` averages to zero on a
# bipartite graph for this circuit, so the current is the edge observable that is checked.
const current = let j = zeros(ComplexF64, 4, 4)
    j[basis(1, 0), basis(0, 1)] = im
    j[basis(0, 1), basis(1, 0)] = -im
    j
end
const density = [0 0; 0 1]

function one_site_operator(matrix, s::Index)
    return operator(TA.project(matrix, (prime(s),), (s,)), (name(prime(s)),), (name(s),))
end
function two_site_operator(matrix, s1::Index, s2::Index)
    codomain, domain = (prime(s1), prime(s2)), (s1, s2)
    return operator(
        TA.project(reshape(matrix, 2, 2, 2, 2), codomain, domain), name.(codomain),
        name.(domain)
    )
end

hopping_gate(θ) = exp(-im * θ * hopping)

expectation(op, ψ) = (conj(ψ) * apply(op, ψ))[] / (conj(ψ) * ψ)[]

# `⟨cᵢ† cⱼ⟩` after the hopping gates `(v1, v2, θ)`, starting from the sites in `occupied` filled.
function reference_correlations(vs, occupied, gates)
    index = Dict(v => i for (i, v) in enumerate(vs))
    C = zeros(ComplexF64, length(vs), length(vs))
    for v in occupied
        C[index[v], index[v]] = 1
    end
    for (v1, v2, θ) in gates
        u = Matrix{ComplexF64}(I, length(vs), length(vs))
        ij = [index[v1], index[v2]]
        u[ij, ij] = exp(-im * θ * [0 1; 1 0])
        C = conj(u) * C * transpose(u)
    end
    return C
end

# Fill the pairs of neighboring sites in `pairs` from the vacuum, apply the hopping gates `hops`,
# and return the densities and the currents of the resulting state, with site sectors of type `S`.
function circuit_observables(S, g, pairs, hops)
    site_axes = Dict(v => Index(gradedrange([S(0) => 1, S(1) => 1])) for v in vertices(g))
    network = tensornetwork(vertices(g)) do v
        return TA.project([1.0 + 0im, 0], (site_axes[v],))
    end
    for edge in edges(g)
        insertlink!(network, edge)
    end
    env = message_environment(one, NormNetwork(network))
    fill_gates = [
        two_site_operator(pair_creation, site_axes[v1], site_axes[v2]) for
            (v1, v2) in pairs
    ]
    hop_gates = [
        two_site_operator(hopping_gate(θ), site_axes[v1], site_axes[v2]) for
            (v1, v2, θ) in hops
    ]
    network, env = apply_operators([fill_gates; hop_gates], network, env)
    ψ = prod(network)
    densities = Dict(
        v => expectation(one_site_operator(density, site_axes[v]), ψ) for v in vertices(g)
    )
    currents = Dict(
        e => expectation(
                two_site_operator(current, site_axes[src(e)], site_axes[dst(e)]),
                ψ
            )
            for e in edges(g)
    )
    return densities, currents
end

@testset "free fermions ($label)" for (label, g, pairs) in (
        ("comb tree", named_comb_tree((3, 2)), [((1, 1), (1, 2)), ((3, 1), (3, 2))]),
        ("cycle", named_cycle_graph(6), [(1, 2), (4, 5)]),
    )
    vs = collect(vertices(g))
    index = Dict(v => i for (i, v) in enumerate(vs))
    hops = [(src(e), dst(e), 0.2 + 0.1 * k) for k in 1:3 for e in edges(g)]
    C = reference_correlations(vs, [v for pair in pairs for v in pair], hops)

    densities, currents = circuit_observables(FermionParity, g, pairs, hops)
    for v in vs
        @test densities[v] ≈ C[index[v], index[v]] atol = 1.0e-8
    end
    for e in edges(g)
        i, j = index[src(e)], index[dst(e)]
        @test currents[e] ≈ im * (C[i, j] - C[j, i]) atol = 1.0e-8
    end

    # Hardcore bosons (a bosonic ℤ₂ grading) go through the same circuit without the fermion
    # signs and come out with different densities on these graphs, so the checks above are
    # sensitive to the signs.
    boson_densities, _ = circuit_observables(Z2Irrep, g, pairs, hops)
    @test maximum(abs(boson_densities[v] - C[index[v], index[v]]) for v in vs) > 0.05
end
