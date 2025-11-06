using DiagonalArrays: δ
using Graphs: edges, ne, nv, vertices
using ITensorBase: Index
using ITensorNetworksNext: contract_network
using ITensorNetworksNext.TensorNetworkGenerators: delta_network, ising_network
using NamedDimsArrays: inds
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset

module TestUtils
    using QuadGK: quadgk
    # Exact critical inverse temperature for 2D square lattice Ising model.
    βc() = 0.5 * log(1 + √2)
    # Exact infinite volume free energy density for 2D square lattice Ising model.
    function ising_free_energy_density(β::Real)
        κ = 2sinh(2β) / cosh(2β)^2
        integrand(θ) = log(0.5 * (1 + sqrt(abs(1 - (κ * sin(θ))^2))))
        integral, _ = quadgk(integrand, 0, π)
        return (-log(2cosh(2β)) - (1 / (2π)) * integral) / β
    end
end

@testset "TensorNetworkGenerators" begin
    @testset "Delta Network" begin
        dims = (3, 3)
        g = named_grid(dims)
        ldict = Dict(e => Index(2) for e in edges(g))
        l(e) = get(() -> ldict[reverse(e)], ldict, e)
        tn = delta_network(l, g)
        @test nv(tn) == 9
        @test ne(tn) == ne(g)
        @test issetequal(vertices(tn), vertices(g))
        @test issetequal(arranged_edges(tn), arranged_edges(g))
        for v in vertices(tn)
            is = l.(incident_edges(g, v))
            @test tn[v] == δ(Tuple(is))
        end
    end
    @testset "Ising Network" begin
        dims = (4, 4)
        β = TestUtils.βc()
        g = named_grid(dims; periodic = true)
        ldict = Dict(e => Index(2) for e in edges(g))
        l(e) = get(() -> ldict[reverse(e)], ldict, e)
        tn = ising_network(l, β, g)
        @test nv(tn) == 16
        @test ne(tn) == ne(g)
        @test issetequal(vertices(tn), vertices(g))
        @test issetequal(arranged_edges(tn), arranged_edges(g))
        for v in vertices(tn)
            is = l.(incident_edges(g, v))
            @test issetequal(is, inds(tn[v]))
            @test tn[v] ≠ δ(Tuple(is))
        end
        # TODO: Use eager contraction sequence finding.
        z = contract_network(tn; alg = "exact")[]
        f = -log(z) / (β * nv(g))
        f_inf = TestUtils.ising_free_energy_density(β)
        @test f ≈ f_inf rtol = 1.0e-1
    end
end
