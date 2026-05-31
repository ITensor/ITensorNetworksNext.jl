using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # `Base.one` and `MatrixAlgebraKit.one!` on `AbstractNamedDimsOperator` are local
    # stand-ins until the upstream `NamedDimsArrays` / `TensorAlgebra` `similar_operator`
    # family lands. Mark the piracy check as broken so Aqua doesn't fail the suite on
    # those expected piracies.
    Aqua.test_all(
        ITensorNetworksNext;
        persistent_tasks = false,
        piracies = (; broken = true)
    )
end
