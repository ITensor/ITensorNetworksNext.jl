using DiagonalArrays: δ
using Graphs: edges, ne, nv, vertices
using ITensorBase: Index
using ITensorNetworksNext: contract_network
using ITensorNetworksNext.TensorNetworkGenerators: delta_network, ising_network
using NamedDimsArrays: inds
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset

!@isdefined(TestUtils) && include("utils.jl")

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
        @testset "1D Ising (periodic = $periodic)" for periodic in (false, true)
            dims = (4,)
            β = 0.4
            g = named_grid(dims; periodic)
            ldict = Dict(e => Index(2) for e in edges(g))
            l(e) = get(() -> ldict[reverse(e)], ldict, e)
            tn = ising_network(l, β, g)
            @test nv(tn) == 4
            @test ne(tn) == ne(g)
            @test issetequal(vertices(tn), vertices(g))
            @test issetequal(arranged_edges(tn), arranged_edges(g))
            for v in vertices(tn)
                is = l.(incident_edges(g, v))
                @test issetequal(is, inds(tn[v]))
                @test tn[v] ≠ δ(Tuple(is))
            end
            z = contract_network(tn)[]
            f = -log(z) / (β * nv(g))
            f_analytic = TestUtils.f_1d_ising(β, 4; periodic)
            @test f ≈ f_analytic
        end
        @testset "2D Ising" begin
            dims = (4, 4)
            β = TestUtils.βc_2d_ising()
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
            z = contract_network(tn)[]
            f = -log(z) / (β * nv(g))
            f_inf = TestUtils.f_2d_ising(β)
            @test f ≈ f_inf rtol = 1.0e-1
        end
    end
end
