using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # Stand-in Base extension on `AbstractNamedDimsOperator` that will move
    # upstream into `NamedDimsArrays.jl` (or its operator extensions).
    # Whitelist it for the piracy check until the upstream PR lands:
    # * `Base.one` on `AbstractNamedDimsOperator` — identity operator,
    #   analog of the existing `Base.sqrt` / `Base.exp` / … extensions
    #   already defined in NDA's `MATRIX_FUNCTIONS` loop.
    Aqua.test_all(
        ITensorNetworksNext;
        persistent_tasks = false,
        piracies = (; treat_as_own = [Base.one])
    )
end
