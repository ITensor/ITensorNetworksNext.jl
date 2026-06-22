using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # Piracy is checked separately as `broken`: `dmrg.jl` temporarily pirates a few
    # `VectorInterface` methods on `AbstractNamedDimsArray` (needed by `KrylovKit.eigsolve`).
    # These are slated to move into `NamedDimsArrays`; drop the `broken` marker once they do.
    Aqua.test_all(ITensorNetworksNext; persistent_tasks = false, piracies = false)
    Aqua.test_piracies(ITensorNetworksNext; broken = true)
end
