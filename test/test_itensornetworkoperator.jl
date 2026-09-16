using DataGraphs: is_vertex_assigned
using Dictionaries: isinsertable, issettable
using Graphs: edges, vertices
using ITensorBase: ITensor, Index, dimnames, inputnames, name, operator, outputnames, state
using ITensorNetworksNext: ITensorNetwork, ITensorNetworkOperator, supportof, tensornetwork
using NamedGraphs: incident_edges, named_path_graph
using Test: @test, @test_throws, @testset

# A bondless operator network on `g`, with one input/output pair per vertex.
function local_operator(g; d = 2)
    inp = Dict(v => Index(d) for v in vertices(g))
    out = Dict(v => Index(d) for v in vertices(g))
    tn = tensornetwork(vertices(g)) do v
        return randn((out[v], inp[v]))
    end
    vs = collect(vertices(g))
    op = operator(tn, [name(out[v]) for v in vs], [name(inp[v]) for v in vs])
    return op, inp, out
end

@testset "`ITensorNetworkOperator`" begin
    @testset "Basics" begin
        g = named_path_graph(3)
        op, inp, out = local_operator(g)

        @test op isa ITensorNetworkOperator

        # The operator shares the graph structure of the network it wraps.
        @test issetequal(vertices(op), vertices(state(op)))
        @test issetequal(edges(op), edges(state(op)))
        @test parent(op) === state(op)

        # The pairing is positional: `outputnames[i]` goes with `inputnames[i]`.
        vs = collect(vertices(g))
        @test outputnames(op) == [name(out[v]) for v in vs]
        @test inputnames(op) == [name(inp[v]) for v in vs]

        @test is_vertex_assigned(op, first(vs))

        # It is a lazy wrapper, so it is neither settable nor insertable.
        @test !issettable(op)
        @test !isinsertable(op)
    end

    @testset "`getindex` wraps the vertex tensor" begin
        g = named_path_graph(3)
        op, inp, out = local_operator(g)

        # Both halves of each pair sit on the same vertex here, so the wrapper carries them.
        for v in vertices(g)
            @test outputnames(op[v]) == [name(out[v])]
            @test inputnames(op[v]) == [name(inp[v])]
            @test state(op[v]) === state(op)[v]
        end

        # `eltype` is the type of the wrapped vertex data.
        @test eltype(op) === typeof(op[first(vertices(g))])
    end

    @testset "a pair may straddle two vertices" begin
        # A swap: the output at vertex 1 is paired with the input at vertex 2, and vice versa.
        a, b = Index(2), Index(2)
        a′, b′ = Index(2), Index(2)
        tn = ITensorNetwork(Dict(1 => randn((a′, a)), 2 => randn((b′, b))))
        op = operator(tn, [name(a′), name(b′)], [name(b), name(a)])

        @test outputnames(op) == [name(a′), name(b′)]
        @test inputnames(op) == [name(b), name(a)]

        # Neither pair is local to a vertex, so each vertex wrapper has an empty pairing and
        # both of its legs stay dangling.
        for v in (1, 2)
            @test isempty(outputnames(op[v]))
            @test isempty(inputnames(op[v]))
            @test issetequal(dimnames(op[v]), dimnames(state(op)[v]))
        end
    end

    @testset "constructor validation" begin
        g = named_path_graph(2)
        op, inp, out = local_operator(g)
        tn = state(op)
        vs = collect(vertices(g))

        # The pairing is a bijection, so the two name lists must have equal length.
        @test_throws ArgumentError operator(tn, [name(out[vs[1]])], [])
        @test_throws ArgumentError operator(
            tn, [name(out[v]) for v in vs], [name(inp[vs[1]])]
        )

        # An operator leg must be dangling: a link name touches two vertices.
        l = Index(2)
        linked = tensornetwork(vertices(g)) do v
            return randn((inp[v], l))
        end
        @test_throws ArgumentError operator(linked, [name(l)], [name(inp[vs[1]])])
    end

    @testset "`supportof`" begin
        g = named_path_graph(3)
        s = Dict(v => Index(2) for v in vertices(g))
        state_tn = tensornetwork(vertices(g)) do v
            is = map(e -> Index(2), incident_edges(g, v))
            return randn((s[v], is...))
        end

        # An operator acting on sites 1 and 3 is supported on exactly those vertices.
        out1, out3 = Index(2), Index(2)
        optn = ITensorNetwork(Dict(1 => randn((out1, s[1])), 3 => randn((out3, s[3]))))
        op = operator(optn, [name(out1), name(out3)], [name(s[1]), name(s[3])])

        @test supportof(state_tn, op) == Set([1, 3])
    end
end
