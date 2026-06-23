import AlgorithmsInterface as AI
using DataGraphs: DataGraphs, DataGraph, edge_data, edge_data_type
using Dictionaries: Dictionary, dictionary, set!
using Graphs: AbstractGraph, dst, edges, has_edge, src, vertices
using ITensorBase: ITensor, Index, inds, name, noprime, prime
using ITensorNetworksNext: ITensorNetworksNext, MessageCache, StopWhenConverged,
    ITensorNetwork, bethe_free_energy, edge_scalar, incoming_messages, linkinds,
    messagecache, region_scalar, subgraph, tensornetwork, vertex_scalar, vertex_scalars
using LinearAlgebra: LinearAlgebra
using NamedGraphs.GraphsExtensions: all_edges, arranged_edges, incident_edges, vertextype
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid, named_path_graph
using NamedGraphs: NamedEdge
using StableRNGs: StableRNG
using Test: @test, @testset

function spin_ice_tensornetwork(g)
    links = DataGraph(g)
    for e in edges(g)
        links[e] = Index(2)
    end

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
    return ITensorNetwork(ts)
end

@testset "Belief propagation" begin
    @testset "`MessageCache`" begin
        @testset "Basics" begin
            dims = (3, 3)
            g = named_grid(dims)

            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))

            tn = tensornetwork(vertices(g)) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(Tuple(is))
            end

            bpc = messagecache(edge -> "$(src(edge)) => $(dst(edge))", all_edges(g))

            @test valtype(bpc) <: String
            @test edge_data_type(bpc) <: String
            @test valtype(bpc) === edge_data_type(bpc)
            @test length(edge_data(bpc)) == 2 * length(edges(g))
            @test bpc[(1, 1) => (1, 2)] == "(1, 1) => (1, 2)"

            # set message
            bpc[(1, 1) => (1, 2)] = "new message"
            @test bpc[(1, 1) => (1, 2)] == "new message"

            pairs = [((1, 2) => (2, 2), "m1"), ((2, 2) => (2, 3), "m2")]

            new_bpc = copyto!(deepcopy(bpc), Dict(pairs))
            @test new_bpc[(1, 1) => (1, 2)] == "new message"
            @test new_bpc[(1, 2) => (2, 2)] == "m1"
            @test new_bpc[(2, 2) => (2, 3)] == "m2"

            new_bpc = copyto!(deepcopy(bpc), dictionary(pairs))
            @test new_bpc[(1, 1) => (1, 2)] == "new message"
            @test new_bpc[(1, 2) => (2, 2)] == "m1"
            @test new_bpc[(2, 2) => (2, 3)] == "m2"

            bpc_dst = messagecache(edge -> "", all_edges(g))

            copyto!(bpc_dst, bpc, [(1, 2) => (2, 2), (2, 2) => (2, 3)])
            @test bpc_dst[(1, 1) => (1, 2)] == ""
            @test bpc_dst[(1, 2) => (2, 2)] == "(1, 2) => (2, 2)"
            @test bpc_dst[(2, 2) => (2, 3)] == "(2, 2) => (2, 3)"
        end
        @testset "Vertex/region scalars" begin
            g = named_path_graph(3)
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))

            tn = tensornetwork(vertices(g)) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(ComplexF32, Tuple(is))
            end

            bpc = messagecache(all_edges(g)) do edge
                return ones(Float64, Tuple(linkinds(tn, edge)))
            end

            # Vertex/edge/region scalars.
            @test vertex_scalar(tn, bpc, 2) isa ComplexF64
            @test edge_scalar(bpc, 1 => 2) isa Float64

            @test region_scalar(tn, bpc, [1]) == vertex_scalar(tn, bpc, 1)
            @test region_scalar(tn, bpc, [2, 3]) == prod(vertex_scalars(tn, bpc, [2, 3]))

            # `incoming_messages` excludes the reverse of the passed edge
            in_msgs = incoming_messages(bpc, 2 => 3)
            @test length(in_msgs) == 1
            @test only(in_msgs) == bpc[1 => 2]

            in_msgs = incoming_messages(bpc, NamedEdge(1 => 2))
            @test length(in_msgs) == 0

            in_msgs = incoming_messages(bpc, NamedEdge(2 => 1))
            @test length(in_msgs) == 1
            @test only(in_msgs) == bpc[3 => 2]
        end

        @testset "subgraph" begin
            g = named_grid((3,))
            l = Dict(e => Index(2) for e in edges(g))
            l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))

            tn = tensornetwork(vertices(g)) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(Tuple(is))
            end
            bpc = messagecache(edge -> ones(Tuple(linkinds(tn, edge))), all_edges(g))

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

            tn = tensornetwork(vertices(g)) do v
                is = map(e -> l[e], incident_edges(g, v))
                return randn(Tuple(is))
            end

            bpc1 = messagecache(edge -> ones(Tuple(linkinds(tn, edge))), all_edges(g))

            bpc2 = copy(bpc1)

            # Identical caches: diff should be ~0.
            @test ITensorNetworksNext.iterate_diff(bpc1, bpc2) ≈ 0.0 atol = 10 * eps()
        end
    end

    @testset "Algorithm" begin
        @testset "$T" for T in (Float32, Float64, ComplexF64, BigFloat)
            rng = StableRNG(123)

            #Chain of tensors
            dims = (2, 1)
            g = DataGraph(named_grid(dims)) # graph to hold the links.
            for edge in edges(g)
                g[edge] = Index(2)
            end

            tensors = map(vertices(g)) do vertex
                is = map(edge -> g[edge], incident_edges(g, vertex))
                return randn(T, Tuple(is))
            end
            tn = ITensorNetwork(tensors)

            messages = Dict(
                edge => ones(T, Tuple(linkinds(tn, edge))) for edge in all_edges(g)
            )

            cache = ITensorNetworksNext.beliefpropagation(
                tn, messages; stopping_criterion = (; maxiter = 1)
            )
            z_bp = exp(bethe_free_energy(tn, cache))
            z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
            @test z_bp ≈ z_exact rtol = eps(real(T))^(1 / 3)

            #Tree of tensors
            dims = (4, 3)
            g = DataGraph(named_comb_tree(dims)) # graph to hold the links.
            for edge in edges(g)
                g[edge] = Index(3)
            end
            tensors = map(vertices(g)) do vertex
                is = map(edge -> g[edge], incident_edges(g, vertex))
                return randn(T, Tuple(is))
            end
            tn = ITensorNetwork(tensors)

            messages = Dict(
                edge => ones(T, Tuple(linkinds(tn, edge))) for edge in all_edges(g)
            )

            cache = ITensorNetworksNext.beliefpropagation(
                tn, messages; stopping_criterion = (; maxiter = 1)
            )
            z_bp = exp(bethe_free_energy(tn, cache))
            z_exact = reduce(*, [tn[v] for v in vertices(g)])[]
            @test z_bp ≈ z_exact rtol = eps(real(T))^(1 / 3)

            #Spin Ice Model (has analytical bp solution given by 1.5^(n^2))
            @testset "Spin Ice Model (analytical)" begin
                for n in (3, 4, 5)
                    dims = (n, n)
                    g = named_grid(dims; periodic = true)
                    tn = spin_ice_tensornetwork(g)

                    messages = Dict(
                        edge => rand(rng, T, Tuple(linkinds(tn, edge)))
                            for edge in all_edges(g)
                    )

                    cache = ITensorNetworksNext.beliefpropagation(
                        tn, messages;
                        stopping_criterion = (; maxiter = 10, tol = 1.0e-10)
                    )

                    z_bp = exp(bethe_free_energy(tn, cache))

                    @test z_bp ≈ 1.5^(n^2)
                end
            end
        end
    end
end
