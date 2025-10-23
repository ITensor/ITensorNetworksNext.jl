using Graphs: edges
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensorBase: Index, ITensor
using ITensorNetworksNext:
    TensorNetwork, linkinds, siteinds, contractnetwork, contraction_sequence, symnameddims, lazy
using TensorOperations: TensorOperations
using Test: @test, @testset

@testset "ContractNetwork" begin
    @testset "Contract Vectors of ITensors" begin
        i, j, k = Index(2), Index(2), Index(5)
        A = ITensor([1.0 1.0; 0.5 1.0], i, j)
        B = ITensor([2.0, 1.0], i)
        C = ITensor([5.0, 1.0], j)
        D = ITensor([-2.0, 3.0, 4.0, 5.0, 1.0], k)

        #@show s1 * s2
        #seq = contraction_sequence([A, B, C, D]; alg = "optimal")
        #@show seq

        #ABCD_1 = contractnetwork([A, B, C, D]; alg = "exact", sequence = "leftassociative")
        # ABCD_2 = contractnetwork([A, B, C, D]; alg = "exact", sequence = "optimal")

        # @test ABCD_1 == ABCD_2
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

        z1 = contractnetwork(tn; alg = "exact", sequence = "optimal")[]
        z2 = contractnetwork(tn; alg = "exact", sequence = "leftassociative")[]

        @test abs(z1 - z2) / abs(z1) <= 1.0e3 * eps(Float64)
    end
end
