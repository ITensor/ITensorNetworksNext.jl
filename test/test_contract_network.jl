using Graphs: edges, vertices
using ITensorBase: Index
using ITensorNetworksNext: contract_network, linkinds, siteinds, tensornetwork
using ITensorNetworksNext.LazyITensors: Greedy, Optimal
using ITensorNetworksNext:
    Exact, LeftAssociative, TensorNetwork, contract_network, linkinds, siteinds, tensornetwork
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using TensorOperations: TensorOperations
using Test: @test, @testset

@testset "contract_network" begin
    orderalg = order_alg -> Exact(; order_alg)

    @testset "Contract Vectors of ITensors" begin
        i, j, k = Index(2), Index(2), Index(5)
        A = [1.0 1.0; 0.5 1.0][i, j]
        B = [2.0, 1.0][i]
        C = [5.0, 1.0][j]
        D = [-2.0, 3.0, 4.0, 5.0, 1.0][k]

        ABCD_1 = contract_network([A, B, C, D]; alg = orderalg(LeftAssociative()))
        ABCD_2 = contract_network([A, B, C, D]; alg = orderalg(Greedy()))
        ABCD_3 = contract_network([A, B, C, D]; alg = orderalg(Optimal()))
        @test ABCD_1 == ABCD_2 == ABCD_3
    end

    @testset "Contract One Dimensional Network" begin
        dims = (4, 4)
        g = named_grid(dims)
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = tensornetwork(vertices(g)) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        z1 = contract_network(tn; alg = orderalg(LeftAssociative()))[]
        z2 = contract_network(tn; alg = orderalg(Greedy()))[]
        z3 = contract_network(tn; alg = orderalg(Optimal()))[]

        @test abs(z1 - z2) / abs(z1) <= 1.0e3 * eps(Float64)
        @test abs(z1 - z3) / abs(z1) <= 1.0e3 * eps(Float64)

        @test z1 ≈ z2
        @test z1 ≈ z3
    end
end
