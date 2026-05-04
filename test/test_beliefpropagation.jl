import AlgorithmsInterface as AI
using DataGraphs: edge_data
using DiagonalArrays: δ
using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, edges, has_edge, src, vertices
using ITensorBase: ITensor, Index, noprime, prime
using ITensorNetworksNext: ITensorNetworksNext, MessageCache, StopWhenConverged,
    TensorNetwork, adapt_messages, edge_scalar, factor, factors, incoming_messages,
    linkinds, map_messages, message, message_type, messages, region_scalar, scalar,
    setmessage!, setmessages!, subgraph, vertex_scalar, vertex_scalars
using LinearAlgebra: LinearAlgebra
using NamedDimsArrays: inds, name
using NamedGraphs.GraphsExtensions: arranged_edges, incident_edges, vertextype
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_path_graph
using Test: @test, @testset

function spin_ice_tensornetwork(g)
    links = Dictionary(
        edges(g),
        [Index(2) for e in edges(g)]
        # [Index(2; tags = "edge " => "e$(src(e))_$(dst(e))") for e in edges(g)]
    )
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))

    ts = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        es = incident_edges(g, v; dir = :in)
        t_data = zeros(Int, 2, 2, 2, 2)
        for (i, j, k, l) in Iterators.product(0:1, 0:1, 0:1, 0:1)
            if i + j + k + l == 2
                t_data[i + 1, j + 1, k + 1, l + 1] = 1
            end
        end
        linkinds = [links[e] for e in es]
        t = t_data[linkinds...]
        set!(ts, v, t)
    end
    return TensorNetwork(g, ts)
end

@testset "Belief propagation" begin
    @testset "`MessageCache`" begin
        @testset "Basics" begin
            dims = (3, 3)
            g = named_grid(dims)
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))

            tn = TensorNetwork(g) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(Tuple(is))
            end

            # By default for graphs, assume factors refers to the vertex data
            @test length(factors(tn)) == 9
            @test factor(tn, (1, 1)) == tn[(1, 1)]

            bpc = MessageCache(tn) do edge
                return "$(src(edge)) => $(dst(edge))"
            end

            @test message_type(bpc) <: String
            @test length(messages(bpc)) == 2 * length(edges(g))
            @test bpc[(1, 1) => (1, 2)] == "(1, 1) => (1, 2)"
            @test message(bpc, (2, 1) => (1, 1)) == "(2, 1) => (1, 1)"

            # set message
            setmessage!(bpc, (1, 1) => (1, 2), "new message")
            @test message(bpc, (1, 1) => (1, 2)) == "new message"

            setmessages!(bpc, Dict(((1, 2) => (2, 2)) => "m1", ((2, 2) => (2, 3)) => "m2"))
            @test message(bpc, (1, 1) => (1, 2)) == "new message"
            @test message(bpc, (1, 2) => (2, 2)) == "m1"
            @test message(bpc, (2, 2) => (2, 3)) == "m2"

            bpc_dst = MessageCache(tn) do edge
                return ""
            end
            setmessages!(bpc_dst, bpc, [(1, 2) => (2, 2), (2, 2) => (2, 3)])
            @test message(bpc_dst, (1, 1) => (1, 2)) == ""
            @test message(bpc, (1, 2) => (2, 2)) == "m1"
            @test message(bpc, (2, 2) => (2, 3)) == "m2"
        end
        @testset "Vertex/region scalars" begin
            g = named_path_graph(3)
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
            tn = TensorNetwork(g) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(ComplexF32, Tuple(is))
            end

            bpc = MessageCache(tn) do edge
                return ones(Float64, Tuple(linkinds(tn, edge)))
            end

            # Vertex/edge/region scalars.
            @test vertex_scalar(tn, bpc, 2) isa ComplexF64
            @test edge_scalar(bpc, 1 => 2) isa Float64

            @test region_scalar(tn, bpc, [1]) == vertex_scalar(tn, bpc, 1)
            @test region_scalar(tn, bpc, [2, 3]) == prod(vertex_scalars(tn, bpc, [2, 3]))

            # `incoming_messages` excludes specified edges.
            in_msgs = incoming_messages(bpc, 2)
            in_msgs_filtered = incoming_messages(
                bpc, 2; ignore_edges = [1 => 2]
            )
            @test length(in_msgs) == 2
            @test length(in_msgs_filtered) == 1
            @test only(in_msgs_filtered) == bpc[3 => 2]

            # `map_messages` and `map_factors` produce independent caches.
            bpc_again = map_messages(identity, bpc)
            @test bpc_again !== bpc
            @test bpc_again == bpc

            bpc_doubled = map_messages(m -> 2 .* m, bpc)
            @test bpc_doubled != bpc
            @test message(bpc_doubled, 1 => 2) ≈ 2 .* message(bpc, 1 => 2)
            @test message(bpc_doubled, 2 => 3) ≈ 2 .* message(bpc, 2 => 3)

            @test adapt_messages(identity, bpc) == bpc
        end

        @testset "subgraph" begin
            g = named_grid((3,))
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
            tn = TensorNetwork(g) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(Tuple(is))
            end
            bpc = MessageCache(tn) do edge
                return ones(Tuple(linkinds(tn, edge)))
            end

            sub_vs = [(1,), (2,)]
            subbpc = subgraph(bpc, sub_vs)
            @test subbpc isa MessageCache
            @test issetequal(vertices(subbpc), sub_vs)
            @test has_edge(subbpc, (1,) => (2,))
        end
        @testset "diff" begin
            g = named_grid((2,))
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
            tn = TensorNetwork(g) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(Tuple(is))
            end

            bpc1 = MessageCache(tn) do edge
                return ones(Tuple(linkinds(tn, edge)))
            end
            bpc2 = MessageCache(tn) do edge
                return ones(Tuple(linkinds(tn, edge)))
            end

            # Identical caches: diff should be ~0.
            @test ITensorNetworksNext.iterate_diff(bpc1, bpc2) ≈ 0.0 atol = 10 * eps()
        end
    end

    @testset "Algorithm" begin
        @testset "$T" for T in (Float32, Float64, ComplexF64, BigFloat)
            #Chain of tensors
            dims = (2, 1)
            g = named_grid(dims)
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))

            tn = TensorNetwork(g) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(T, Tuple(is))
            end

            bpc = MessageCache(tn) do edge
                return ones(T, Tuple(linkinds(tn, edge)))
            end
            bpc = ITensorNetworksNext.beliefpropagation(tn, bpc; maxiter = 1)
            z_bp = scalar(tn, bpc)
            z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
            @test z_bp ≈ z_exact

            #Tree of tensors
            dims = (4, 3)
            g = named_comb_tree(dims)
            l = Dict(e => Index(3) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
            tn = TensorNetwork(g) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(T, Tuple(is))
            end

            bpc = MessageCache(tn) do edge
                return ones(T, Tuple(linkinds(tn, edge)))
            end
            bpc = ITensorNetworksNext.beliefpropagation(tn, bpc; maxiter = 1)
            z_bp = scalar(tn, bpc)
            z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
            @test z_bp ≈ z_exact

            #Spin Ice Model (has analytical bp solution given by 1.5^(n^2))
            @testset "Spin Ice Model (analytical)" begin
                for n in (3, 4, 5)
                    dims = (n, n)
                    g = named_grid(dims; periodic = true)
                    tn = spin_ice_tensornetwork(g)

                    bpc = ITensorNetworksNext.MessageCache(tn) do edge
                        # Use `rand` so messages have positive elements.
                        return rand(T, Tuple(linkinds(tn, edge)))
                    end

                    stopping_criterion = StopWhenConverged(tol = 1.0e-10)

                    bpc =
                        ITensorNetworksNext.beliefpropagation(
                        tn,
                        bpc;
                        maxiter = 10,
                        stopping_criterion
                    )

                    z_bp = scalar(tn, bpc)

                    @test z_bp ≈ 1.5^(n^2)
                end
            end
        end
    end
end
