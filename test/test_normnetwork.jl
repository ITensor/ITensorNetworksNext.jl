using DataGraphs: is_vertex_assigned
using Dictionaries: isinsertable, issettable
using Graphs: edges, vertices
using ITensorBase: ITensor, Index, IndexName, conj, inds, name, setname, uniquename
using ITensorNetworksNext: BraView, Greedy, ITensorNetwork, KetView, NormNetwork, braname,
    bratensor, conj_bratensor, contraction_tree, flatten_network, indmap, kettensor,
    normnetwork, prod_tensors, tensornetwork
using LinearAlgebra: norm
using NamedGraphs: NamedEdge, incident_edges, named_grid, named_path_graph
using Test: @test, @test_throws, @testset

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

# A norm network carries no tensor at a vertex, so contracting it goes through the flattened
# network, whose two layers are separate operands.
contract_norm(nn) = prod_tensors(flatten_network(nn), Greedy())[]

@testset "`NormNetwork`" begin
    @testset "Basics" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        nn = NormNetwork(tn)

        # `normnetwork` is the public constructor and agrees with `NormNetwork`.
        @test normnetwork(tn) isa NormNetwork
        @test nn isa NormNetwork

        # The norm network shares the graph structure of the underlying network.
        @test issetequal(vertices(nn), vertices(tn))
        @test issetequal(edges(nn), edges(tn))

        # The double-layer tensor at a vertex has no representation, so neither indexing nor
        # `eltype` is defined.
        @test_throws ErrorException nn[1]
        @test_throws ErrorException eltype(nn)

        # Vertex data is assigned wherever the underlying network is.
        @test is_vertex_assigned(nn, 1)

        # The norm network is neither settable nor insertable (it is a lazy view).
        @test !issettable(nn)
        @test !isinsertable(nn)
    end

    @testset "kettensor / bratensor / conj_bratensor and the name map" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        nn = NormNetwork(tn)

        # `kettensor` returns the underlying tensor untouched.
        @test kettensor(nn, 2) === tn[2]

        # Site indices appear in a single tensor, so they are *not* renamed: the ket and
        # bra layers share them (they get contracted, forming the physical overlap).
        sname = name(s[2])
        @test braname(nn, sname) == sname
        @test sname in name.(inds(kettensor(nn, 2)))
        @test sname in name.(inds(conj_bratensor(nn, 2)))

        # Link indices are shared by two tensors, so they *are* renamed in the bra layer
        # to keep the two layers' bonds distinct.
        lname = name(l[NamedEdge(1 => 2)])
        @test braname(nn, lname) != lname
        @test lname in name.(inds(kettensor(nn, 2)))
        @test !(lname in name.(inds(conj_bratensor(nn, 2))))
        @test braname(nn, lname) in name.(inds(conj_bratensor(nn, 2)))

        # `bra` is the elementwise conjugate of `conj_bratensor` and carries the same indices.
        @test inds(bratensor(nn, 2)) == inds(conj_bratensor(nn, 2))

        # `indmap` conjugates an index and renames it according to the name map.
        ind = only(i for i in inds(kettensor(nn, 2)) if name(i) == lname)
        @test name(indmap(nn, ind)) == braname(nn, name(ind))
        @test indmap(nn, ind) == setname(conj(ind), braname(nn, name(ind)))

        # Querying the name map with an index name absent from the network errors.
        @test_throws ErrorException braname(nn, name(Index(2)))
    end

    @testset "custom name map" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)

        # A user-supplied map dictates the bra-layer name for each link.
        custom = map(uniquename, keys(tn.dimname_vertices))
        nn = normnetwork(tn, custom)

        lname = name(l[NamedEdge(1 => 2)])
        @test braname(nn, lname) == custom[lname]
        @test braname(nn, lname) in name.(inds(conj_bratensor(nn, 2)))
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
            @test kv[v] === kettensor(nn, v)
            @test inds(bv[v]) == inds(bratensor(nn, v))
        end

        @test is_vertex_assigned(kv, 1)
        @test is_vertex_assigned(bv, 1)

        # Views inherit the (non-)mutability of their parent norm network.
        @test !issettable(kv)
        @test !isinsertable(kv)
        @test !issettable(bv)
        @test !isinsertable(bv)
    end

    @testset "`flatten_network`" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        nn = NormNetwork(tn)
        flat = flatten_network(nn)

        # One vertex per layer, so the vertex count doubles and each carries its layer's tensor.
        @test length(vertices(flat)) == 2 * length(vertices(nn))
        # Each vertex's two layers stay adjacent, in ket-then-bra order.
        @test collect(vertices(flat)) ==
            collect(Iterators.flatten(((v, :ket), (v, :bra)) for v in vertices(nn)))
        @test flat[(2, :ket)] === kettensor(nn, 2)
        @test flat[(2, :bra)] == bratensor(nn, 2)

        # A norm network carries no tensor at a vertex, so it has to be flattened first.
        @test_throws ErrorException prod_tensors(nn, Greedy())

        # A tree is keyed by the vertices of the network it was built over, so one built over
        # the flattened network replays against it.
        tree = contraction_tree(flat)
        @test prod_tensors(flat, tree)[] ≈ prod_tensors(flat, Greedy())[]
    end

    @testset "contraction / physics" begin
        @testset "single normalized tensor contracts to 1" begin
            s = Index(3)
            v = randn(s)
            v = v / norm(v)
            tn = ITensorNetwork(Dict(1 => v))
            nn = NormNetwork(tn)

            # ⟨ψ|ψ⟩ for a single normalized site tensor is 1.
            @test contract_norm(nn) ≈ 1
        end

        @testset "$T" for T in (Float64, ComplexF64)
            g = named_grid((2, 2))
            tn, l, s = random_state(T, g)

            # The norm network contracts to ⟨tn|tn⟩ = ‖prod(tn)‖², a real nonnegative number.
            z = contract_norm(NormNetwork(tn))
            @test z ≈ norm(prod(tn))^2
            @test imag(z) ≈ 0 atol = 1.0e-12 * abs(z)
            @test real(z) > 0

            # Rescaling a single tensor by 1/√z normalizes the state, so ⟨tn|tn⟩ = 1.
            tn[first(vertices(tn))] = tn[first(vertices(tn))] / sqrt(real(z))
            @test contract_norm(NormNetwork(tn)) ≈ 1

            # The contracted norm does not depend on the chosen bra-layer name map.
            custom = map(uniquename, keys(tn.dimname_vertices))
            @test contract_norm(normnetwork(tn, custom)) ≈ contract_norm(NormNetwork(tn))
        end
    end
end
