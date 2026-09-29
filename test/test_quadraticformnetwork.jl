using DataGraphs: is_vertex_assigned
using Dictionaries: isinsertable, issettable
using Graphs: edges, vertices
using ITensorBase: ITensor, Index, IndexName, conj, inds, inputnames, name, names, operator,
    outputnames, rename, state, uniquename
using ITensorNetworksNext: ITensorNetworksNext, BraView, ITensorNetwork, NormNetwork,
    QuadraticFormGramian, QuadraticFormNetwork, braname, branetwork, bratensor,
    conj_bratensor, contract_network, ketnetwork, kettensor, operatornetwork,
    operatortensor, quadraticformnetwork, tensornetwork
using LinearAlgebra: I, norm
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

# Build a bondless operator layer on the vertices of `g`, with `f(v)` supplying the matrix
# acting on the site index `s[v]`.
function product_operator(f, g, s; d = 2)
    out = Dict(v => Index(d) for v in vertices(g))
    tn = tensornetwork(vertices(g)) do v
        return ITensor(f(v), (out[v], s[v]))
    end
    vs = collect(vertices(g))
    return operator(tn, [name(out[v]) for v in vs], [name(s[v]) for v in vs])
end

identity_operator(g, s; d = 2) = product_operator(v -> Matrix(1.0I, d, d), g, s; d)

@testset "`QuadraticFormNetwork`" begin
    @testset "Basics" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        op = identity_operator(g, s)
        qf = QuadraticFormNetwork(tn, op)

        # `quadraticformnetwork` is the public constructor and agrees with the type.
        @test quadraticformnetwork(tn, op) isa QuadraticFormNetwork
        @test qf isa QuadraticFormNetwork

        # The quadratic form shares the graph structure of the ket layer.
        @test issetequal(vertices(qf), vertices(tn))
        @test issetequal(edges(qf), edges(tn))

        # `eltype` is the type of the (lazy triple-layer) vertex data.
        @test eltype(qf) === typeof(qf[1])

        # Vertex data is assigned wherever both layers are.
        @test is_vertex_assigned(qf, 1)

        # The quadratic form is neither settable nor insertable (it is a lazy view).
        @test !issettable(qf)
        @test !isinsertable(qf)

        # The operator layer must cover every vertex of the ket layer.
        g4 = named_path_graph(4)
        _, _, s4 = random_state(Float64, g4)
        @test_throws ErrorException QuadraticFormNetwork(tn, identity_operator(g4, s4))
    end

    @testset "`QuadraticFormGramian`" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        op = identity_operator(g, s)
        qf = QuadraticFormNetwork(tn, op)
        gram = qf[2]

        @test gram isa QuadraticFormGramian
        @test eltype(qf) === typeof(gram)
        @test kettensor(gram) === tn[2]
        # The Gramian holds the operator at its vertex, with that vertex's pairing.
        @test state(gram.operator) === state(op)[2]
        @test outputnames(gram.operator) == outputnames(op[2])
        @test inputnames(gram.operator) == [name(s[2])]
        @test keys(ITensorNetworksNext.layertensors(gram)) == (:ket, :operator, :bra)
        @test contract_network([gram]) ≈
            kettensor(gram) * operatortensor(gram) * bratensor(gram)
        # A Gramian's indices are those no two of its layers share.
        @test issetequal(
            inds(gram), inds(kettensor(gram) * operatortensor(gram) * bratensor(gram))
        )
        @test names(gram) == name.(inds(gram))
        @test axes(gram) == Tuple(inds(gram))
    end

    @testset "operator input outside the ket network" begin
        g = named_path_graph(2)
        tn, l, s = random_state(Float64, g)
        stray = Index(2)
        out1, out2 = Index(2), Index(2)
        optn = ITensorNetwork(Dict(1 => randn((out1, stray)), 2 => randn((out2, s[2]))))
        op = operator(optn, [name(out1), name(out2)], [name(stray), name(s[2])])
        @test_throws ErrorException QuadraticFormNetwork(tn, op)
    end

    @testset "layer tensors and the name map" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        op = identity_operator(g, s)
        qf = QuadraticFormNetwork(tn, op)

        # `kettensor` returns the underlying tensor untouched.
        @test kettensor(qf[2]) === tn[2]

        # Unlike the norm network, the site indices *are* renamed in the bra layer: the
        # operator sits between the two layers, so they no longer contract directly.
        sname = name(s[2])
        @test braname(qf, sname) != sname
        @test sname in name.(inds(kettensor(qf[2])))
        @test !(sname in name.(inds(conj_bratensor(qf[2]))))
        @test braname(qf, sname) in name.(inds(conj_bratensor(qf[2])))

        # Link indices are shared by two tensors, so they are renamed in the bra layer to
        # keep the two layers' bonds distinct.
        lname = name(l[NamedEdge(1 => 2)])
        @test braname(qf, lname) != lname
        @test lname in name.(inds(kettensor(qf[2])))
        @test !(lname in name.(inds(conj_bratensor(qf[2]))))
        @test braname(qf, lname) in name.(inds(conj_bratensor(qf[2])))

        # The operator's input name meets the ket and its output name is renamed to meet
        # the bra.
        o = operatortensor(qf[2])
        @test sname in names(o)
        @test braname(qf, sname) in names(o)
        @test inputnames(op) == [name(s[v]) for v in vertices(g)]

        # `bratensor` is the elementwise conjugate of `conj_bratensor` and carries the same
        # indices.
        @test inds(bratensor(qf[2])) == inds(conj_bratensor(qf[2]))

        # Querying the name map with an index name absent from the ket layer errors.
        @test_throws ErrorException braname(qf, name(Index(2)))
    end

    @testset "custom name map" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        op = identity_operator(g, s)

        # A user-supplied map dictates the bra-layer name for each renamed index.
        custom = map(uniquename, keys(tn.dimname_vertices))
        qf = quadraticformnetwork(tn, op, custom)

        lname = name(l[NamedEdge(1 => 2)])
        @test braname(qf, lname) == custom[lname]
        @test braname(qf, lname) in name.(inds(conj_bratensor(qf[2])))

        sname = name(s[2])
        @test braname(qf, sname) == custom[sname]
    end

    @testset "`ketnetwork` / `operatornetwork` / `branetwork`" begin
        g = named_path_graph(3)
        tn, l, s = random_state(Float64, g)
        op = identity_operator(g, s)
        qf = QuadraticFormNetwork(tn, op)

        # The ket and operator layers are the networks the quadratic form was built from.
        @test ketnetwork(qf) === tn
        @test operatornetwork(qf) === op

        # The bra layer is not stored, so it is a view sharing the ket layer's graph structure.
        bv = branetwork(qf)
        @test bv isa BraView
        @test issetequal(vertices(bv), vertices(tn))
        @test issetequal(edges(bv), edges(tn))
        @test !issettable(bv)
        @test !isinsertable(bv)
        @test is_vertex_assigned(bv, 1)
        for v in vertices(tn)
            @test inds(bv[v]) == inds(bratensor(qf[v]))
        end
    end

    @testset "contraction / physics" begin
        @testset "identity operator layer reproduces the norm network" begin
            g = named_grid((2, 2))
            tn, l, s = random_state(Float64, g)
            op = identity_operator(g, s)

            @test contract_network(QuadraticFormNetwork(tn, op))[] ≈
                contract_network(NormNetwork(tn))[]
        end

        @testset "$T" for T in (Float64, ComplexF64)
            g = named_grid((2, 2))
            tn, l, s = random_state(T, g)

            # A product of on-site matrices, contracted densely for the reference value.
            mats = Dict(v => randn(T, 2, 2) for v in vertices(g))
            op = product_operator(v -> mats[v], g, s)
            qf = QuadraticFormNetwork(tn, op)

            # ⟨ψ|O|ψ⟩ built by applying each on-site matrix to the dense ket and
            # overlapping with the dense bra.
            psi = prod(tn)
            ket = psi
            for v in vertices(g)
                out = Index(2)
                gate = ITensor(mats[v], (out, s[v]))
                ket = rename(gate * ket, name(out) => name(s[v]))
            end
            @test contract_network(qf)[] ≈ (conj(psi) * ket)[]
        end

        @testset "scaled identity" begin
            g = named_path_graph(3)
            tn, l, s = random_state(Float64, g)
            op = product_operator(v -> 2.0 * Matrix(1.0I, 2, 2), g, s)

            # A factor of 2 at each of the three vertices scales the norm by 2³.
            @test contract_network(QuadraticFormNetwork(tn, op))[] ≈
                8 * contract_network(NormNetwork(tn))[]
        end
    end
end
