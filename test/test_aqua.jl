using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ITensorNetworksNext; persistent_tasks = false)
end
