using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # `Base.one(::AbstractNamedDimsOperator)` is a local stand-in until the upstream
    # `NamedDimsArrays` / `TensorAlgebra` `id_operator` / `similar_operator` family
    # lands. Mark the piracy check as broken so Aqua doesn't fail the suite on that
    # expected piracy.
    Aqua.test_all(
        ITensorNetworksNext;
        persistent_tasks = false,
        piracies = (; broken = true)
    )
end
