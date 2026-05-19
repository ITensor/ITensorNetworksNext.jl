using Aqua: Aqua
using ITensorNetworksNext: ITensorNetworksNext
using MatrixAlgebraKit: MatrixAlgebraKit
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # Stand-in Base / MAK extensions on `AbstractNamedDimsArray` /
    # `AbstractNamedDimsOperator` that will move upstream into
    # `NamedDimsArrays.jl` (or its operator extensions). Whitelist them
    # for the piracy check until the upstream PRs land:
    # * `MAK.inv_regularized` — N-d pseudo-inverse for named arrays.
    # * `Base.one` on `AbstractNamedDimsOperator` — identity operator,
    #   analog of the existing `Base.sqrt` / `Base.exp` / … extensions
    #   already defined in NDA's `MATRIX_FUNCTIONS` loop.
    Aqua.test_all(
        ITensorNetworksNext;
        persistent_tasks = false,
        piracies = (; treat_as_own = [MatrixAlgebraKit.inv_regularized, Base.one])
    )
end
