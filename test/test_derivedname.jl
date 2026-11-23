using ITensorBase: Index, name, plev, prime, tags
using ITensorNetworksNext: derivedname
using Test: @test, @testset

@testset "`derivedname`" begin
    n = prime(name(Index(2; tags = "Link")))

    # The derived name depends only on the name and key, keeps the decoration and is tagged
    # with the key.
    @test derivedname(n, :bra) == derivedname(n, :bra)
    @test derivedname(n, :bra) != n
    @test derivedname(n, :bra) != derivedname(n, :other)
    @test derivedname(n, :bra) != prime(n)
    @test tags(derivedname(n, :bra)) == merge(tags(n), Dict("bra" => ""))
    @test plev(derivedname(n, :bra)) == plev(n)
end
