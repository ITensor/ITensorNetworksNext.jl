using Base.Broadcast: materialize
using DataGraphs: is_vertex_assigned
using Dictionaries: isinsertable, issettable
using Graphs: edges, vertices
using ITensorBase:
    ITensor, Index, IndexName, LazyITensor, conj, inds, name, setname, uniquename
using ITensorNetworksNext: BraView, ITensorNetwork, KetView, NormNetwork, bra, conjbra,
    indmap, ket, namemap, normnetwork, tensornetwork
using LinearAlgebra: norm
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid, named_path_graph
using NamedGraphs: NamedEdge
using Test: @test, @test_throws, @testset

# Contract a (possibly double-layer) network into a single tensor by multiplying all
# of its vertex tensors together. For a `NormNetwork` the vertex data are lazy products
# `ket * conj(bra)`, so the result is a lazy expression that we materialize.
contract(tn) = materialize(prod(tn))

# Build a random `ITensorNetwork` state on the graph `g` with site dimension `d` and
# bond dimension `χ`.
function random_state(::Type{T}, g; d = 2, χ = 2) where {T}
    l = Dict(e => Index(χ) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    s = Dict(v => Index(d) for v in vertices(g))
    tn = tensornetwork(vertices(g)) do v
        is = map(e -> l[e], incident_edges(g, v))
        return randn(T, (s[v], is...))
    end
    return tn, l, s
end

@testset "`NormNetwork`" begin
    @testset "Basics" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        nn = NormNetwork(tn)

        # `normnetwork` is the public constructor and agrees with `NormNetwork`.
        @test normnetwork(tn) isa NormNetwork
        @test nn isa NormNetwork

        # The underlying tensor network is accessible and is the same object.
        @test tensornetwork(nn) === tn

        # The norm network shares the graph structure of the underlying network.
        @test issetequal(vertices(nn), vertices(tn))
        @test issetequal(edges(nn), edges(tn))

        # `eltype` is the type of the (lazy double-layer) vertex data.
        @test eltype(nn) === typeof(nn[first(vertices(nn))])

        # Vertex data is assigned wherever the underlying network is.
        @test is_vertex_assigned(nn, 1)

        # The norm network is neither settable nor insertable (it is a lazy view).
        @test !issettable(nn)
        @test !isinsertable(nn)
    end

    @testset "ket / bra / conjbra and the name map" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        nn = NormNetwork(tn)

        # `ket` returns the underlying tensor untouched.
        @test ket(nn, 2) === tn[2]

        # Site indices appear in a single tensor, so they are *not* renamed: the ket and
        # bra layers share them (they get contracted, forming the physical overlap).
        sname = name(s[2])
        @test namemap(nn, sname) == sname
        @test sname in name.(inds(ket(nn, 2)))
        @test sname in name.(inds(conjbra(nn, 2)))

        # Link indices are shared by two tensors, so they *are* renamed in the bra layer
        # to keep the two layers' bonds distinct.
        lname = name(l[NamedEdge(1 => 2)])
        @test namemap(nn, lname) != lname
        @test lname in name.(inds(ket(nn, 2)))
        @test !(lname in name.(inds(conjbra(nn, 2))))
        @test namemap(nn, lname) in name.(inds(conjbra(nn, 2)))

        # `bra` is the elementwise conjugate of `conjbra` and carries the same indices.
        @test inds(bra(nn, 2)) == inds(conjbra(nn, 2))

        # `indmap` conjugates an index and renames it according to the name map.
        ind = only(i for i in inds(ket(nn, 2)) if name(i) == lname)
        @test name(indmap(nn, ind)) == namemap(nn, name(ind))
        @test indmap(nn, ind) == setname(conj(ind), namemap(nn, name(ind)))

        # Querying the name map with an index name absent from the network errors.
        @test_throws ErrorException namemap(nn, name(Index(2)))
    end

    @testset "custom name map" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)

        # A user-supplied map dictates the bra-layer name for each link.
        custom = map(uniquename, keys(tn.dimname_vertices))
        nn = normnetwork(tn, custom)

        lname = name(l[NamedEdge(1 => 2)])
        @test namemap(nn, lname) == custom[lname]
        @test namemap(nn, lname) in name.(inds(conjbra(nn, 2)))
    end

    @testset "`KetView` / `BraView`" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        nn = NormNetwork(tn)

        kv = KetView(nn)
        bv = BraView(nn)

        # Views share the graph structure of the underlying network.
        @test issetequal(vertices(kv), vertices(tn))
        @test issetequal(vertices(bv), vertices(tn))
        @test issetequal(edges(kv), edges(tn))
        @test issetequal(edges(bv), edges(tn))

        # The ket view exposes the bare ket tensors; the bra view exposes the bra tensors.
        for v in vertices(tn)
            @test kv[v] === ket(nn, v)
            @test inds(bv[v]) == inds(bra(nn, v))
        end

        @test is_vertex_assigned(kv, 1)
        @test is_vertex_assigned(bv, 1)

        # Views inherit the (non-)mutability of their parent norm network.
        @test !issettable(kv)
        @test !isinsertable(kv)
        @test !issettable(bv)
        @test !isinsertable(bv)
    end

    @testset "contraction / physics" begin
        @testset "single normalized tensor contracts to 1" begin
            s = Index(3)
            v = randn(s)
            v = v / norm(v)
            tn = ITensorNetwork(Dict(1 => v))
            nn = NormNetwork(tn)

            # ⟨ψ|ψ⟩ for a single normalized site tensor is 1.
            @test contract(nn)[] ≈ 1
        end

        @testset "$T" for T in (Float64, ComplexF64)
            g = named_grid((2, 2))
            tn, l, s = random_state(T, g)

            # The norm network contracts to ⟨tn|tn⟩ = ‖prod(tn)‖², a real nonnegative number.
            z = contract(NormNetwork(tn))[]
            @test z ≈ norm(prod(tn))^2
            @test imag(z) ≈ 0 atol = 1.0e-12 * abs(z)
            @test real(z) > 0

            # Rescaling a single tensor by 1/√z normalizes the state, so ⟨tn|tn⟩ = 1.
            tn[first(vertices(tn))] = tn[first(vertices(tn))] / sqrt(real(z))
            @test contract(NormNetwork(tn))[] ≈ 1

            # The contracted norm does not depend on the chosen bra-layer name map.
            custom = map(uniquename, keys(tn.dimname_vertices))
            @test contract(normnetwork(tn, custom))[] ≈ contract(NormNetwork(tn))[]
        end
    end
end
