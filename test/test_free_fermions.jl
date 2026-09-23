using GradedArrays: U1
using Graphs: dst, edges, src, vertices
using ITensorBase: Index, apply, inds, operator, prime
using ITensorNetworksNext: NormNetwork, apply_operator, apply_operators, insertlink!,
    message_environment, tensornetwork
using NamedGraphs: named_comb_tree, named_cycle_graph
using TensorAlgebra: project, project_aux
using TensorKitSectors: FermionNumber
using Test: @test, @testset

# Free fermions on a tree with a branching vertex and on a loop, where the fermion signs change
# the observables, evolved in imaginary time bond by bond and compared with the exact Slater
# determinant. The circuit is not truncated, so the comparison is exact.

const cdag_matrix = Float64[0 0; 1 0]
const c_matrix = Float64[0 1; 0 0]

# `project_aux` derives an auxiliary leg carrying the charge of a charge-shifting operator, so
# `c†` conserves charge with that leg left dangling, and `c†ᵢ cⱼ` pairs a `c†` and a `c` over one
# shared such leg.
function cdag(s::Index)
    return operator(project_aux(cdag_matrix, (prime(s),), (s,)), [prime(s)], [s])
end
function cdag_c(si::Index, sj::Index)
    a = project_aux(cdag_matrix, (prime(si),), (si,))
    aux = last(inds(a))
    b = project(reshape(c_matrix, 2, 2, 1), (prime(sj),), (sj, aux))
    return operator(a, [prime(si)], [si]) * operator(b, [prime(sj)], [sj])
end
hopping(si::Index, sj::Index) = cdag_c(si, sj) + cdag_c(sj, si)
number(s::Index) = operator(project([0 0; 0 1], (prime(s),), (s,)), [prime(s)], [s])

expectation(o, ψ) = (conj(ψ) * apply(o, ψ))[] / (conj(ψ) * ψ)[]

# Densities on the vertices and hoppings on the edges from the full state `ψ`.
function full_state_correlators(g, sites, ψ)
    densities = Dict(v => expectation(number(sites[v]), ψ) for v in vertices(g))
    hoppings = Dict(
        e => expectation(hopping(sites[src(e)], sites[dst(e)]), ψ) for e in edges(g)
    )
    return densities, hoppings
end

# Densities and hoppings of free fermions created on the sites in `occupied` and evolved by
# `exp(-τ h)` on the bonds `(v1, v2, τ)`. The state stays a Slater determinant with orbitals `Φ`,
# each gate multiplies `Φ` by `exp(-τ h)` on its two sites, and `⟨cᵢ† cⱼ⟩ = (Φ (Φᵀ Φ)⁻¹ Φᵀ)ᵢⱼ`.
function free_fermion_correlators(g, occupied, bonds)
    index = Dict(v => i for (i, v) in enumerate(vertices(g)))
    Φ = zeros(length(index), length(occupied))
    for (k, v) in enumerate(occupied)
        Φ[index[v], k] = 1
    end
    for (v1, v2, τ) in bonds
        ij = [index[v1], index[v2]]
        Φ[ij, :] = exp(-τ * [0 1; 1 0]) * Φ[ij, :]
    end
    C = Φ * ((Φ' * Φ) \ Φ')
    densities = Dict(v => C[index[v], index[v]] for v in vertices(g))
    hoppings = Dict(
        e => C[index[src(e)], index[dst(e)]] + C[index[dst(e)], index[src(e)]] for
            e in edges(g)
    )
    return densities, hoppings
end

# Create a fermion on each site in `occupied`, then apply `exp(-τ h)` on the bonds `(v1, v2, τ)`
# by belief-propagation simple update. Returns the network and its site indices.
function bp_evolve(sector, g, occupied, bonds)
    sites = Dict(v => Index([sector(0) => 1, sector(1) => 1]) for v in vertices(g))
    ψ = tensornetwork(v -> ones(sites[v]), vertices(g))
    for e in edges(g)
        insertlink!(ψ, e)
    end
    env = message_environment(one, NormNetwork(ψ))
    for v in occupied
        ψ, env = apply_operator(cdag(sites[v]), ψ, env)
    end
    gates = [exp(-τ * hopping(sites[v1], sites[v2])) for (v1, v2, τ) in bonds]
    ψ, env = apply_operators(gates, ψ, env)
    return ψ, sites
end

# An even number of fermions on the cycle, where an odd number would be indistinguishable from
# hardcore bosons.
@testset "free fermions ($label)" for (label, g, occupied) in (
        ("comb tree", named_comb_tree((3, 2)), [(2, 1), (2, 2)]),
        ("cycle", named_cycle_graph(6), [1, 2, 4, 5]),
    )
    bonds = [(src(e), dst(e), 0.5) for _ in 1:3 for e in edges(g)]
    densities, hoppings = free_fermion_correlators(g, occupied, bonds)

    ψ, sites = bp_evolve(FermionNumber, g, occupied, bonds)
    bp_densities, bp_hoppings = full_state_correlators(g, sites, prod(ψ))
    for v in vertices(g)
        @test bp_densities[v] ≈ densities[v] atol = 1.0e-8
    end
    for e in edges(g)
        @test bp_hoppings[e] ≈ hoppings[e] atol = 1.0e-8
    end

    # Hardcore bosons go through the same circuit without the fermion signs and come out
    # different on these graphs, so the checks above are sensitive to the signs.
    ψ_boson, sites_boson = bp_evolve(U1, g, occupied, bonds)
    boson_densities, boson_hoppings = full_state_correlators(g, sites_boson, prod(ψ_boson))
    @test max(
        maximum(abs(boson_densities[v] - densities[v]) for v in vertices(g)),
        maximum(abs(boson_hoppings[e] - hoppings[e]) for e in edges(g))
    ) > 0.05
end
