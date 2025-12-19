import AlgorithmsInterface as AI
using ITensorNetworksNext: EigsolveRegion, dmrg, select_algorithm
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @testset

@testset "select_algorithm(dmrg, ...)" begin
    operator = "operator"
    init = "init"
    nsweeps = 3
    regions = ["region1", "region2"]
    maxdim = [10, 20]
    cutoff = 1.0e-7
    algorithm = select_algorithm(dmrg, operator, init; nsweeps, regions, maxdim, cutoff)
    @test algorithm isa AIE.NestedAlgorithm
    @test length(algorithm.algorithms) == nsweeps

    maxdims = [10, 20, 20]
    cutoffs = [1.0e-7, 1.0e-7, 1.0e-7]
    algorithm′ = AIE.nested_algorithm(nsweeps) do i
        return AIE.nested_algorithm(length(regions)) do j
            return EigsolveRegion(
                regions[j];
                maxdim = maxdims[i],
                cutoff = cutoffs[i],
            )
        end
    end
    for i in 1:nsweeps
        for j in 1:length(regions)
            @test algorithm.algorithms[i].algorithms[j] ==
                algorithm′.algorithms[i].algorithms[j]
        end
    end
end
