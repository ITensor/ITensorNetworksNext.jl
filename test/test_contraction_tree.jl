using AbstractTrees: AbstractTrees, Leaves
using Graphs: edges, vertices
using ITensorBase: Index
using ITensorNetworksNext: Branch, ContractionTree, Greedy, contraction_tree, isleaf,
    left_associative_tree, prod_tensors, tensornetwork
using NamedGraphs: incident_edges, named_grid
using OMEinsumContractionOrders: GreedyMethod, TreeSA
using Test: @test, @testset

@testset "ContractionTree" begin
    @testset "leaves and branches" begin
        l = ContractionTree(:a)
        @test isleaf(l)
        @test AbstractTrees.nodevalue(l) == :a
        @test AbstractTrees.children(l) == ()

        # (a, (b, c))
        t = ContractionTree(
            ContractionTree(:a),
            ContractionTree(ContractionTree(:b), ContractionTree(:c))
        )
        @test !isleaf(t)
        @test length(AbstractTrees.children(t)) == 2
        @test t[1] == ContractionTree(:a)
        @test isleaf(t[1])
        @test !isleaf(t[2])
        @test AbstractTrees.nodevalue(t[2][1]) == :b
        @test AbstractTrees.nodevalue(t[2][2]) == :c
    end

    @testset "a branch comes from `Branch`, never from a bare tuple" begin
        a, b = ContractionTree(1), ContractionTree(2)
        @test ContractionTree(Branch(a, b)) == ContractionTree(a, b)
        @test !isleaf(ContractionTree(Branch(a, b)))

        # A bare 2-tuple is a label, even when its entries are themselves trees. This is what
        # keeps a tuple-valued vertex label from being read as children.
        @test isleaf(ContractionTree((a, b)))
        @test AbstractTrees.nodevalue(ContractionTree((a, b))) == (a, b)
    end

    @testset "equality and hashing" begin
        t1 = ContractionTree(ContractionTree(:a), ContractionTree(:b))
        t2 = ContractionTree(ContractionTree(:a), ContractionTree(:b))
        t3 = ContractionTree(ContractionTree(:b), ContractionTree(:a))
        @test t1 == t2
        @test hash(t1) == hash(t2)
        @test t1 != t3
        # a leaf whose label collides structurally is not equal to a branch
        @test ContractionTree(:a) !=
            ContractionTree(ContractionTree(:a), ContractionTree(:a))
    end

    @testset "arbitrary label types, including Vector-valued leaves" begin
        # The ambiguity that nested `Vector` cannot represent: a leaf whose label is itself a
        # vector. Leaf/branch is a type distinction, so this is unambiguous.
        l = ContractionTree([1, 2])
        @test isleaf(l)
        @test AbstractTrees.nodevalue(l) == [1, 2]

        t = ContractionTree(ContractionTree([1, 2]), ContractionTree([3, 4]))
        @test !isleaf(t)
        @test AbstractTrees.nodevalue(t[1]) == [1, 2]
        @test AbstractTrees.nodevalue(t[2]) == [3, 4]
    end

    @testset "tuple-valued vertex labels are leaves, not branches" begin
        # A `Tuple`-valued label (e.g. grid coordinates) must not be mistaken for a branch.
        # Leaves and branches are separate types, so a label can never be read as children.
        l = ContractionTree((1, 2))
        @test isleaf(l)
        @test AbstractTrees.nodevalue(l) == (1, 2)

        t = ContractionTree(ContractionTree((1, 2)), ContractionTree((3, 4)))
        @test !isleaf(t)
        @test isleaf(t[1])
        @test AbstractTrees.nodevalue(t[1]) == (1, 2)
        @test AbstractTrees.nodevalue(t[2]) == (3, 4)
    end

    @testset "show" begin
        t = ContractionTree(
            ContractionTree(1),
            ContractionTree(ContractionTree(2), ContractionTree(3))
        )
        @test sprint(show, t) == "(1, (2, 3))"
        @test sprint(show, ContractionTree(:a)) == ":a"
    end

    @testset "`prod_tensors` follows the tree" begin
        i, j, k = Index(2), Index(2), Index(5)
        A = [1.0 1.0; 0.5 1.0][i, j]
        B = [2.0, 1.0][i]
        C = [5.0, 1.0][j]
        D = [-2.0, 3.0, 4.0, 5.0, 1.0][k]
        ts = [A, B, C, D]

        leaves = map(ContractionTree, 1:4)
        left = left_associative_tree(1:4)
        @test left == reduce(ContractionTree, leaves)
        balanced = ContractionTree(
            ContractionTree(leaves[1], leaves[2]), ContractionTree(leaves[3], leaves[4])
        )
        @test prod_tensors(ts, left) ≈ ((A * B) * C) * D
        @test prod_tensors(ts, balanced) ≈ (A * B) * (C * D)
    end

    @testset "`contraction_tree` over a grid" begin
        g = named_grid((2, 3))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = tensornetwork(vertices(g)) do v
            return randn(Tuple(map(e -> l[e], incident_edges(g, v))))
        end

        for alg in (Greedy(), GreedyMethod(), TreeSA())
            tree = contraction_tree(tn; alg)
            # Every vertex is contracted exactly once, under its own key.
            @test length(collect(Leaves(tree))) == length(vertices(g))
            @test issetequal(map(AbstractTrees.nodevalue, Leaves(tree)), vertices(g))
            @test prod_tensors(tn, tree)[] ≈ prod_tensors(tn, Greedy())[]
        end

        # A tree depends only on the network's shape, so it replays against a second network
        # built over the same graph and indices.
        tree = contraction_tree(tn; alg = Greedy())
        tn2 = tensornetwork(vertices(g)) do v
            return randn(Tuple(map(e -> l[e], incident_edges(g, v))))
        end
        @test prod_tensors(tn2, tree)[] ≈ prod_tensors(tn2, Greedy())[]
    end
end
