using Base.Broadcast: materialize
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, Prod, lazy
using NamedDimsArrays: NamedDimsArray, inds, nameddims
using TermInterface:
    arguments, arity, children, head, iscall, isexpr, maketerm, operation, sorted_arguments
using Test: @test, @test_throws, @testset
using WrappedUnions: unwrap

@testset "LazyNamedDimsArrays" begin
    @testset "Basics" begin
        a1 = nameddims(randn(2, 2), (:i, :j))
        a2 = nameddims(randn(2, 2), (:j, :k))
        a3 = nameddims(randn(2, 2), (:k, :l))
        l1, l2, l3 = lazy.((a1, a2, a3))
        for li in (l1, l2, l3)
            @test li isa LazyNamedDimsArray
            @test unwrap(li) isa NamedDimsArray
            @test inds(li) == inds(unwrap(li))
            @test copy(li) == unwrap(li)
            @test materialize(li) == unwrap(li)
        end
        l = l1 * l2 * l3
        @test copy(l) ≈ a1 * a2 * a3
        @test materialize(l) ≈ a1 * a2 * a3
        @test issetequal(inds(l), symdiff(inds.((a1, a2, a3))...))
        @test unwrap(l) isa Prod
        @test unwrap(l).factors == [l1 * l2, l3]
    end

    @testset "TermInterface" begin
        a1 = nameddims(randn(2, 2), (:i, :j))
        a2 = nameddims(randn(2, 2), (:j, :k))
        a3 = nameddims(randn(2, 2), (:k, :l))
        l1, l2, l3 = lazy.((a1, a2, a3))

        @test_throws ErrorException arguments(l1)
        @test_throws ErrorException arity(l1)
        @test_throws ErrorException children(l1)
        @test_throws ErrorException head(l1)
        @test !iscall(l1)
        @test !isexpr(l1)
        @test_throws ErrorException operation(l1)
        @test_throws ErrorException sorted_arguments(l1)

        l = l1 * l2 * l3
        @test arguments(l) == [l1 * l2, l3]
        @test arity(l) == 2
        @test children(l) == [l1 * l2, l3]
        @test head(l) ≡ prod
        @test iscall(l)
        @test isexpr(l)
        @test l == maketerm(LazyNamedDimsArray, prod, [l1 * l2, l3], nothing)
        @test operation(l) ≡ prod
        @test sorted_arguments(l) == [l1 * l2, l3]
    end
end
