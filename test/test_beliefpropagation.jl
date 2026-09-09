import AlgorithmsInterface as AI
using Base.Broadcast: materialize
using DataGraphs: DataGraphs, DataGraph, edge_data, edge_data_type
using Dictionaries: Dictionary, dictionary, set!
using GradedArrays: U1, gradedrange
using Graphs: AbstractGraph, add_vertex!, dst, edges, has_edge, has_vertex, nv, rem_edge!,
    src, vertices
using ITensorBase: Greedy, ITensor, Index, inds, name, noprime, outputnames, prime
using ITensorNetworksNext: ITensorNetworksNext, Exact, ITensorNetwork, MessageCache,
    NormNetwork, SimpleMessageUpdate, StopWhenConverged, beliefpropagation,
    bethe_free_energy, contract_network, contraction_order, edge_scalar, factor_tensors,
    incoming_messages, insertlink!, linkinds, message_environment, messagecache,
    region_scalar, subgraph, tensornetwork, updated_message, vertex_scalar, vertex_scalars
using LinearAlgebra: LinearAlgebra
using NamedGraphs: NamedEdge, all_edges, incident_edges, named_comb_tree, named_grid,
    named_path_graph, vertextype
using StableRNGs: StableRNG
using TensorKitSectors: FermionParity
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

# Records how many operands each `contract_network` call is given, then orders them greedily.
struct RecordOperands
    counts::Vector{Int}
end
function ITensorNetworksNext.contraction_order(alg::RecordOperands, tn)
    push!(alg.counts, length(tn))
    return contraction_order(tn; alg = Greedy())
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
        @testset "Graphs.jl mutation" begin
            g = named_path_graph(3)
            bpc = messagecache(edge -> "$(src(edge)) => $(dst(edge))", all_edges(g))

            @test add_vertex!(bpc, 4)
            @test has_vertex(bpc, 4)
            @test !add_vertex!(bpc, 4)
            @test nv(bpc) == 4

            nmessages = length(edge_data(bpc))
            @test rem_edge!(bpc, 1 => 2)
            @test !has_edge(bpc, 1 => 2)
            @test length(edge_data(bpc)) == nmessages - 1
            @test !rem_edge!(bpc, 1 => 2)
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

            tn = tensornetwork(vertices(g)) do vertex
                is = map(edge -> g[edge], incident_edges(g, vertex))
                return randn(T, Tuple(is))
            end

            messages = Dict(
                edge => ones(T, Tuple(linkinds(tn, edge))) for edge in all_edges(g)
            )

            cache = beliefpropagation(
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
            tn = tensornetwork(vertices(g)) do vertex
                is = map(edge -> g[edge], incident_edges(g, vertex))
                return randn(T, Tuple(is))
            end

            messages = Dict(
                edge => ones(T, Tuple(linkinds(tn, edge))) for edge in all_edges(g)
            )

            cache = beliefpropagation(
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

                    cache = beliefpropagation(
                        tn, messages;
                        stopping_criterion = (; maxiter = 10, tol = 1.0e-10)
                    )

                    z_bp = exp(bethe_free_energy(tn, cache))

                    @test z_bp ≈ 1.5^(n^2)
                end
            end
        end
    end

    @testset "NormNetwork (operator-valued messages)" begin
        site_ranges = (
            "U1" => gradedrange([U1(0) => 1, U1(1) => 1]),
            "FermionParity" =>
                gradedrange([FermionParity(0) => 1, FermionParity(1) => 1]),
        )
        @testset "$label, T=$T" for (label, site_range) in site_ranges,
                T in (Float64, ComplexF64)

            rng = StableRNG(123)
            g = named_path_graph(4)
            site_axes = Dict(v => Index(site_range) for v in vertices(g))
            network = tensornetwork(vertices(g)) do v
                return randn(rng, T, (site_axes[v],))
            end
            for edge in edges(g)
                insertlink!(network, edge)
            end
            nn = NormNetwork(network)

            cache = beliefpropagation(
                nn, message_environment(one, nn);
                stopping_criterion = (; maxiter = 20, tol = 1.0e-10)
            )

            # Messages stay operator-valued end to end (a plain message has no output names).
            @test all(msg -> !isempty(outputnames(msg)), edge_data(cache))

            # Belief propagation is exact on a tree, including on the fermionic norm network.
            ket = prod(network)
            z_exact = (ket * conj(ket))[]
            z_bp = exp(bethe_free_energy(nn, cache))
            @test z_bp ≈ z_exact rtol = eps(real(T))^(1 / 3)
        end
    end

    @testset "Doubled-vertex contraction operands" begin
        site_ranges = (
            "plain" => 2,
            "U1" => gradedrange([U1(0) => 1, U1(1) => 1]),
        )
        @testset "$label" for (label, site_range) in site_ranges
            rng = StableRNG(1234)
            g = named_grid((3, 3))
            network = tensornetwork(vertices(g)) do v
                return randn(rng, (Index(site_range),))
            end
            for edge in edges(g)
                insertlink!(network, edge)
            end
            nn = NormNetwork(network)
            v = (2, 2)

            # A doubled vertex splits into its two layers, and the split is faithful.
            @test length(factor_tensors(nn, v)) == 2
            @test prod(factor_tensors(nn, v)) ≈ materialize(nn[v])
            # A single-layer network's factor is a single operand.
            @test factor_tensors(network, v) == [network[v]]

            # The message update passes the layers to `contract_network` as separate operands, so
            # the contraction order can interleave the incoming messages between them, and the
            # result matches contracting the doubled vertex as one operand.
            counts = Int[]
            algorithm = SimpleMessageUpdate(;
                contraction_alg = Exact(; order_alg = RecordOperands(counts))
            )
            cache = message_environment(one, nn)
            edge = NamedEdge(v => (2, 3))
            messages = collect(incoming_messages(cache, edge))
            message = updated_message(algorithm, cache, nn, edge)
            # `v` has degree 4, so 3 incoming messages plus the ket and bra layers.
            @test only(counts) == 5
            @test message ≈ contract_network([messages; [nn[v]]])
        end
    end
end
