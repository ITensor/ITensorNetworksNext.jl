using AbstractTrees: AbstractTrees, print_tree, printnode
using Base.Broadcast: materialize
using ITensorBase: @names, ITensor, denamed, dimnames, inds, nameddims, namedoneto
using ITensorNetworksNext.LazyITensors:
    LazyITensor, Mul, SymbolicITensor, ismul, lazy, substitute, symnameddims
using TermInterface: arguments, arity, children, head, iscall, isexpr, maketerm, operation,
    sorted_arguments, sorted_children
using Test: @test, @test_broken, @test_throws, @testset
using WrappedUnions: unwrap

@testset "LazyITensors" begin
    @testset "Basics" begin
        i, j, k, l = namedoneto.(2, (:i, :j, :k, :l))
        a1 = randn(i, j)
        a2 = randn(j, k)
        a3 = randn(k, l)
        l1, l2, l3 = lazy.((a1, a2, a3))
        for li in (l1, l2, l3)
            @test li isa LazyITensor
            @test unwrap(li) isa ITensor
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
        # Show-string format depends on how `Index` names are displayed; not load-bearing.
        @test_broken sprint(printnode, l1) == "[:i, :j]"
        @test_broken sprint(print_tree, l1) == "[:i, :j]\n"

        l = l1 * l2 * l3
        @test arguments(l) == [l1 * l2, l3]
        @test arity(l) == 2
        @test children(l) == [l1 * l2, l3]
        @test head(l) ≡ *
        @test iscall(l)
        @test isexpr(l)
        @test l == maketerm(LazyITensor, *, [l1 * l2, l3], nothing)
        @test operation(l) ≡ *
        @test sorted_arguments(l) == [l1 * l2, l3]
        @test sorted_children(l) == [l1 * l2, l3]
        @test AbstractTrees.children(l) == [l1 * l2, l3]
        @test AbstractTrees.nodevalue(l) ≡ *
        # Show-string format depends on how `Index` names are displayed; not load-bearing.
        @test_broken sprint(show, l) == "(([:i, :j] * [:j, :k]) * [:k, :l])"
        @test_broken sprint(printnode, l) == "(([:i, :j] * [:j, :k]) * [:k, :l])"
        @test_broken sprint(print_tree, l) ==
            "(([:i, :j] * [:j, :k]) * [:k, :l])\n" *
            "├─ ([:i, :j] * [:j, :k])\n" *
            "│  ├─ [:i, :j]\n│  └─ [:j, :k]\n" *
            "└─ [:k, :l]\n"
    end

    @testset "symnameddims" begin
        a1, a2, a3 = symnameddims.((:a1, :a2, :a3))
        @test a1 isa LazyITensor
        @test unwrap(a1) isa SymbolicITensor
        @test unwrap(a1) == SymbolicITensor(:a1, ())
        @test isequal(unwrap(a1), SymbolicITensor(:a1, ()))
        @test inds(a1) == ()
        @test isempty(dimnames(a1))

        ex = a1 * a2 * a3
        @test copy(ex) == ex
        @test arguments(ex) == [a1 * a2, a3]
        @test operation(ex) ≡ *
        @test sprint(show, ex) == "((a1 * a2) * a3)"
    end

    @testset "substitute" begin
        s = symnameddims.((:a1, :a2, :a3))
        i = @names i[1:4]
        a = (randn(2, 2)[i[1], i[2]], randn(2, 2)[i[2], i[3]], randn(2, 2)[i[3], i[4]])
        l = lazy.(a)

        seq = s[1] * (s[2] * s[3])
        net = substitute(seq, s .=> l)
        @test net == l[1] * (l[2] * l[3])
        @test arguments(net) == [l[1], l[2] * l[3]]
    end
end
