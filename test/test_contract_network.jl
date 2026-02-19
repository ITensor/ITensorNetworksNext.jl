using Graphs: edges
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensorBase: Index
using ITensorNetworksNext: TensorNetwork, linkinds, siteinds, contract_network
using TensorOperations: TensorOperations
using Test: @test, @testset
using BackendSelection: @Algorithm_str, Algorithm

@testset "contract_network" begin
    orderalg = alg -> Algorithm"exact"(; order_alg = Algorithm(alg))

    @testset "Contract Vectors of ITensors" begin
        i, j, k = Index(2), Index(2), Index(5)
        A = [1.0 1.0; 0.5 1.0][i, j]
        B = [2.0, 1.0][i]
        C = [5.0, 1.0][j]
        D = [-2.0, 3.0, 4.0, 5.0, 1.0][k]

        ABCD_1 = contract_network([A, B, C, D]; alg = orderalg("left_associative"))
        ABCD_2 = contract_network([A, B, C, D]; alg = orderalg("eager"))
        ABCD_3 = contract_network([A, B, C, D]; alg = orderalg("optimal"))
        @test ABCD_1 == ABCD_2 == ABCD_3
    end

    @testset "Contract One Dimensional Network" begin
        dims = (4, 4)
        g = named_grid(dims)
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        z1 = contract_network(tn; alg = orderalg("left_associative"))[]
        z2 = contract_network(tn; alg = orderalg("eager"))[]
        z3 = contract_network(tn; alg = orderalg("optimal"))[]

        @test abs(z1 - z2) / abs(z1) <= 1.0e3 * eps(Float64)
        @test abs(z1 - z3) / abs(z1) <= 1.0e3 * eps(Float64)
    end
end
