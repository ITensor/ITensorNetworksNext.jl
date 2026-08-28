using Graphs: edges, vertices
using ITensorBase:
    Greedy, Index, NamedTensorOperator, inputnames, operator, outputnames, state
using ITensorNetworksNext: Exact, ITensorNetwork, LeftAssociative, contract_network,
    linkinds, siteinds, tensornetwork
using NamedGraphs: incident_edges, named_grid
using OMEinsumContractionOrders: ExhaustiveSearch, GreedyMethod, TreeSA
using Test: @test, @testset

@testset "contract_network" begin
    orderalg = order_alg -> Exact(; order_alg)

    @testset "Contract Vectors of ITensors" begin
        i, j, k = Index(2), Index(2), Index(5)
        A = [1.0 1.0; 0.5 1.0][i, j]
        B = [2.0, 1.0][i]
        C = [5.0, 1.0][j]
        D = [-2.0, 3.0, 4.0, 5.0, 1.0][k]

        ABCD_1 = contract_network([A, B, C, D]; alg = orderalg(LeftAssociative()))
        ABCD_2 = contract_network([A, B, C, D]; alg = orderalg(Greedy()))
        ABCD_3 = contract_network([A, B, C, D]; alg = orderalg(ExhaustiveSearch()))
        ABCD_4 = contract_network([A, B, C, D]; alg = orderalg(GreedyMethod()))
        ABCD_5 = contract_network([A, B, C, D]; alg = orderalg(TreeSA()))
        @test ABCD_1 == ABCD_2 == ABCD_3
        @test ABCD_1 ≈ ABCD_4
        @test ABCD_1 ≈ ABCD_5
    end

    @testset "Contract One Dimensional Network" begin
        dims = (4, 4)
        g = named_grid(dims)
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = tensornetwork(vertices(g)) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        z1 = contract_network(tn; alg = orderalg(LeftAssociative()))[]
        z2 = contract_network(tn; alg = orderalg(Greedy()))[]
        z3 = contract_network(tn; alg = orderalg(ExhaustiveSearch()))[]
        z4 = contract_network(tn; alg = orderalg(GreedyMethod()))[]
        z5 = contract_network(tn; alg = orderalg(TreeSA()))[]

        @test abs(z1 - z2) / abs(z1) <= 1.0e3 * eps(Float64)
        @test abs(z1 - z3) / abs(z1) <= 1.0e3 * eps(Float64)

        @test z1 ≈ z2
        @test z1 ≈ z3
        @test z1 ≈ z4
        @test z1 ≈ z5
    end

    @testset "Contract network with operators" begin
        i, j, k = Index(2), Index(2), Index(2)
        o = operator(randn(2, 2), (i,), (j,))     # output i, input j

        # A network mixing an operator with plain tensors previously threw a `convert`
        # `MethodError`; it now contracts, stays an operator, and matches the binary product.
        t = randn(2, 2)[j, k]
        r = contract_network([o, t])
        @test r isa NamedTensorOperator
        @test state(r) ≈ state(o * t)
        @test outputnames(r) == outputnames(o * t)
        @test inputnames(r) == inputnames(o * t)

        # An all-operator network is likewise preserved.
        o2 = operator(randn(2, 2), (j,), (k,))
        r2 = contract_network([o, o2])
        @test r2 isa NamedTensorOperator
        @test state(r2) ≈ state(o * o2)

        # A fully-contracted operator network reads out as a scalar via `[]`.
        f = randn(2, 2)[i, j]
        @test contract_network([o, f])[] ≈ (o * f)[]

        # An all-plain network is unaffected: it is not promoted to an operator.
        a = randn(2, 2)[i, j]
        b = randn(2, 2)[j, k]
        @test !(contract_network([a, b]) isa NamedTensorOperator)

        # Pairing is order-independent: a branching network with a surviving output/input pair
        # matches the binary product under any fold order (greedy contraction included).
        ip, mp, m, x = Index(2), Index(2), Index(2), Index(2)
        op = operator(randn(2, 2, 2, 2), (ip, mp), (i, m))
        u = randn(2, 2)[mp, x]
        w = randn(2, 2)[m, x]
        rb = contract_network([op, u, w])
        @test outputnames(rb) == outputnames((op * u) * w) == outputnames(op * (u * w))
        @test inputnames(rb) == inputnames((op * u) * w) == inputnames(op * (u * w))
    end
end
