using ITensorNetworksNext: ITensorNetworksNext
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ITensorNetworksNext)
end
