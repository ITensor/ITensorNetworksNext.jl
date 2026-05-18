using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using MatrixAlgebraKit: MatrixAlgebraKit
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # `MatrixAlgebraKit.inv_regularized` is locally extended for
    # `AbstractNamedDimsArray` as a stand-in until the corresponding method
    # moves into `NamedDimsArrays.jl`. Whitelist it for the piracy check.
    Aqua.test_all(
        ITensorNetworksNext;
        persistent_tasks = false,
        piracies = (; treat_as_own = [MatrixAlgebraKit.inv_regularized])
    )
end
