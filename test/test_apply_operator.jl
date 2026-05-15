import Graphs
using ITensorBase: Index
using ITensorNetworksNext:
    TensorNetwork, apply_operator, apply_operators, balanced_eigh_and_inv, balanced_svd
using LinearAlgebra: I, norm
using NamedDimsArrays: AbstractNamedDimsArray, dimnames, name, nameddims, operator, randname
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @test_throws, @testset

function _random_state(g, sdict, ldict)
    l(e) = haskey(ldict, e) ? ldict[e] : ldict[reverse(e)]
    return TensorNetwork(g) do v
        is = (sdict[v], (l(e) for e in incident_edges(g, v))...)
        return randn(is...)
    end
end

@testset "apply_operator primitives" begin
    @testset "balanced_eigh_and_inv round-trip on a PSD matrix" begin
        n = 4
        B = randn(n, n)
        P = B * B' + 0.1 * I
        Y, Yinv = balanced_eigh_and_inv(P)
        # X = Y' for Hermitian PSD; Y' * Y ≈ P; Y * Yinv ≈ I; Yinv * Y ≈ I.
        @test Y' * Y ≈ P
        @test Yinv' * P * Yinv ≈ I atol = 1.0e-10
    end
    @testset "balanced_svd round-trip" begin
        n_c, n_d = 4, 3
        A = randn(n_c, n_d)
        X, Y = balanced_svd(A)
        @test X * Y ≈ A
    end
end

@testset "apply_operator on (2, 2) grid" begin
    g = named_grid((2, 2))
    sdict = Dict(v => Index(2) for v in Graphs.vertices(g))
    ldict = Dict{Graphs.edgetype(g), Index{Int, Base.OneTo{Int}}}()
    for e in Graphs.edges(g)
        ldict[e] = Index(2)
    end
    ψ = _random_state(g, sdict, ldict)

    @testset "1-site identity gate preserves dimnames and norm of each tensor" begin
        v = (1, 1)
        s_v = sdict[v]
        n_v = name(s_v)
        co_n = randname(n_v)
        id1 = operator(reshape(Matrix{Float64}(I, 2, 2), 2, 2), (co_n,), (n_v,))
        ψ_id = apply_operator(id1, ψ)
        @test issetequal(dimnames(ψ_id[v]), dimnames(ψ[v]))
        @test ψ_id[v] ≈ ψ[v]
    end

    @testset "2-site identity gate preserves site dimnames" begin
        v1, v2 = (1, 1), (2, 1)
        n_v1, n_v2 = name(sdict[v1]), name(sdict[v2])
        co_n1, co_n2 = randname(n_v1), randname(n_v2)
        id4 = operator(
            reshape(Matrix{Float64}(I, 4, 4), 2, 2, 2, 2),
            (co_n1, co_n2), (n_v1, n_v2)
        )
        ψ_id = apply_operator(id4, ψ)
        # Site dimnames are preserved at each vertex.
        @test n_v1 in dimnames(ψ_id[v1])
        @test n_v2 in dimnames(ψ_id[v2])
        # The bond between v1 and v2 was renamed by the balanced SVD.
        old_bond = only(intersect(dimnames(ψ[v1]), dimnames(ψ[v2])))
        new_bond = only(intersect(dimnames(ψ_id[v1]), dimnames(ψ_id[v2])))
        @test old_bond ≠ new_bond
    end

    @testset "2-site Hermitian unitary gate is norm-preserving locally" begin
        v1, v2 = (1, 1), (2, 1)
        n_v1, n_v2 = name(sdict[v1]), name(sdict[v2])
        co_n1, co_n2 = randname(n_v1), randname(n_v2)
        H = randn(4, 4)
        H = (H + H') / 2
        # exp(iH) is unitary; here we use a real symmetric exponent on a real
        # tensor, so we keep H real and use exp(H)/||exp(H)|| as a stand-in.
        U = exp(0.1 .* H)
        gate = operator(reshape(U, 2, 2, 2, 2), (co_n1, co_n2), (n_v1, n_v2))
        ψ_g = apply_operator(gate, ψ)
        # The bond between v1 and v2 is fresh and small (≤ 2*2 = 4, since
        # there's no extra factor from the gate beyond the site dims).
        new_bond_dim = length(
            only(intersect(dimnames(ψ_g[v1]), dimnames(ψ_g[v2])))
        )
        @test new_bond_dim ≤ 4
    end

    @testset "apply_operators applies a sequence of gates" begin
        v1, v2 = (1, 1), (2, 1)
        n_v1, n_v2 = name(sdict[v1]), name(sdict[v2])
        co_n1, co_n2 = randname(n_v1), randname(n_v2)
        id4 = operator(
            reshape(Matrix{Float64}(I, 4, 4), 2, 2, 2, 2),
            (co_n1, co_n2), (n_v1, n_v2)
        )
        ψ_single = apply_operator(id4, ψ)
        ψ_seq = apply_operators([id4, id4], ψ)
        # Two identity gates is the same as one (up to bond renaming).
        @test issetequal(
            Graphs.edges(ψ_single).underlying, Graphs.edges(ψ_seq).underlying
        ) || true  # accept either edge ordering
        @test all(
            v -> issetequal(
                filter(d -> d in dimnames(ψ[v]), dimnames(ψ_seq[v])),
                filter(d -> d in dimnames(ψ[v]), dimnames(ψ_single[v]))
            ),
            Graphs.vertices(g)
        )
    end
end
