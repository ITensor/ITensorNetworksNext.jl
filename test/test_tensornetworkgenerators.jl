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
    βc_2d_ising(elt::Type{<:Number} = Float64) = elt(log(1 + √2) / 2)
    # Exact infinite volume free energy density for 2D square lattice Ising model.
    function f_2d_ising(β::Real; J::Real = one(β))
        κ = 2sinh(2β * J) / cosh(2β * J)^2
        integrand(θ) = log((1 + √(abs(1 - (κ * sin(θ))^2))) / 2)
        integral, _ = quadgk(integrand, 0, π)
        return (-log(2cosh(2β * J)) - (1 / (2π)) * integral) / β
    end
    function f_1d_ising(β::Real; J::Real = one(β), h::Real = zero(β))
        λ⁺ = exp(β * J) * (cosh(β * h) + √(sinh(β * h)^2 + exp(-4β * J)))
        return -(log(λ⁺) / β)
    end
    function f_1d_ising(β::Real, N::Integer; periodic::Bool = true, kwargs...)
        return if periodic
            f_1d_ising_periodic(β, N; kwargs...)
        else
            f_1d_ising_open(β, N; kwargs...)
        end
    end
    function f_1d_ising_periodic(β::Real, N::Integer; J::Real = one(β), h::Real = zero(β))
        r = √(sinh(β * h)^2 + exp(-4β * J))
        λ⁺ = exp(β * J) * (cosh(β * h) + r)
        λ⁻ = exp(β * J) * (cosh(β * h) - r)
        Z = λ⁺^N + λ⁻^N
        return -(log(Z) / (β * N))
    end
    function f_1d_ising_open(β::Real, N::Integer; J::Real = one(β), h::Real = zero(β))
        isone(N) && return 2cosh(β * h)
        T = [
            exp(β * (J + h)) exp(-β * J);
            exp(-β * J) exp(β * (J - h));
        ]
        b = [exp(β * h / 2), exp(-β * h / 2)]
        Z = (b' * (T^(N - 1)) * b)[]
        return -(log(Z) / (β * N))
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
