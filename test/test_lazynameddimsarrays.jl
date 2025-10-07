using AbstractTrees: AbstractTrees, print_tree, printnode
using Base.Broadcast: materialize
using ITensorNetworksNext.LazyNamedDimsArrays:
    LazyNamedDimsArray, Mul, ismul, lazy, symnameddims
using ITensorNetworksNext.SymbolicArrays: SymbolicArray
using NamedDimsArrays: NamedDimsArray, dename, dimnames, inds, nameddims
using TermInterface:
    arguments,
    arity,
    children,
    head,
    iscall,
    isexpr,
    maketerm,
    operation,
    sorted_arguments,
    sorted_children
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
        @test unwrap(l) isa Mul
        @test ismul(unwrap(l))
        @test unwrap(l).arguments == [l1 * l2, l3]
        # TermInterface.jl
        @test operation(unwrap(l)) ≡ *
        @test arguments(unwrap(l)) == [l1 * l2, l3]
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
        @test_throws ErrorException sorted_children(l1)
        @test AbstractTrees.children(l1) ≡ ()
        @test AbstractTrees.nodevalue(l1) ≡ a1
        @test sprint(show, l1) == sprint(show, a1)
        # TODO: Fix this test, it is basically correct but the type parameters
        # print in a different way.
        # @test sprint(show, MIME"text/plain"(), l1) ==
        #     replace(sprint(show, MIME"text/plain"(), a1), "NamedDimsArray" => "LazyNamedDimsArray")
        @test sprint(printnode, l1) == "[:i, :j]"
        @test sprint(print_tree, l1) == "[:i, :j]\n"

        l = l1 * l2 * l3
        @test arguments(l) == [l1 * l2, l3]
        @test arity(l) == 2
        @test children(l) == [l1 * l2, l3]
        @test head(l) ≡ *
        @test iscall(l)
        @test isexpr(l)
        @test l == maketerm(LazyNamedDimsArray, *, [l1 * l2, l3], nothing)
        @test operation(l) ≡ *
        @test sorted_arguments(l) == [l1 * l2, l3]
        @test sorted_children(l) == [l1 * l2, l3]
        @test AbstractTrees.children(l) == [l1 * l2, l3]
        @test AbstractTrees.nodevalue(l) ≡ *
        @test sprint(show, l) == "(([:i, :j] * [:j, :k]) * [:k, :l])"
        @test sprint(show, MIME"text/plain"(), l) == "(([:i, :j] * [:j, :k]) * [:k, :l])"
        @test sprint(printnode, l) == "(([:i, :j] * [:j, :k]) * [:k, :l])"
        @test sprint(print_tree, l) ==
            "(([:i, :j] * [:j, :k]) * [:k, :l])\n├─ ([:i, :j] * [:j, :k])\n│  ├─ [:i, :j]\n│  └─ [:j, :k]\n└─ [:k, :l]\n"
    end

    @testset "symnameddims" begin
        a = symnameddims(:a)
        b = symnameddims(:b)
        c = symnameddims(:c)
        @test a isa LazyNamedDimsArray
        @test unwrap(a) isa NamedDimsArray
        @test dename(a) isa SymbolicArray
        @test dename(unwrap(a)) isa SymbolicArray
        @test dename(unwrap(a)) == SymbolicArray(:a)
        @test inds(a) == ()
        @test dimnames(a) == ()

        ex = a * b * c
        @test copy(ex) == ex
        @test arguments(ex) == [a * b, c]
        @test operation(ex) ≡ *
        @test sprint(show, ex) == "((a[] * b[]) * c[])"
        @test sprint(show, MIME"text/plain"(), ex) == "((a[] * b[]) * c[])"
    end
end
