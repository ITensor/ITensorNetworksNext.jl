import AlgorithmsInterface as AI
import ITensorNetworksNext.AlgorithmsInterfaceExtensions as AIE
using AbstractTrees: AbstractTrees
using BackendSelection: @Algorithm_str, Algorithm
using DataGraphs: vertex_data
using Dictionaries: Dictionary
using Graphs: Graphs, AbstractEdge, dst, edges, has_edge, ne, nv, src, vertices
using ITensorBase: ITensor, Index
using ITensorNetworksNext: BeliefPropagationCache, EigsolveRegion, ITensorNetworksNext,
    TensorNetwork, contract_network, dmrg, factor, factor_type, factors, linkinds, message,
    message_type, messages, scalar
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray, LazyNamedDimsArrays, Mul,
    SymbolicArray, ismul, lazy, parenttype, substitute, symnameddims
using ITensorNetworksNext.TensorNetworkGenerators: ising_network
using NamedDimsArrays: AbstractNamedDimsArray, NamedDimsArray, denamed, dimnames, inds,
    nameddims, namedoneto
using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: GraphsExtensions, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using TermInterface: arguments, head, iscall, isexpr, operation
using Test: @test, @test_throws, @testset
using WrappedUnions: unwrap

# Type definitions used by some tests below; must be at file scope.
struct _DummyNonIter <: AIE.NonIterativeAlgorithm end
struct _DummyProblem <: AIE.Problem end

@testset "test_new.jl" begin
    # ---------------------------------------------------------------------------
    # AbstractTensorNetwork: iteration / keys / eltype / is_directed / show
    # ---------------------------------------------------------------------------
    @testset "AbstractTensorNetwork interface" begin
        g = named_grid((2, 2))
        s = Dict(v => Index(2) for v in vertices(g))
        tn = TensorNetwork(g) do v
            return randn(s[v])
        end

        # `iterate` works (delegates to `vertex_data`).
        @test !isempty(collect(tn))
        # `keys` returns vertices.
        @test issetequal(keys(tn), vertices(tn))
        # `eltype` matches the eltype of the vertex data.
        @test eltype(tn) === eltype(vertex_data(tn))
        # `is_directed` is `false` for AbstractTensorNetwork.
        @test !Graphs.is_directed(typeof(tn))

        # `show` MIME and default both succeed and mention vertices/edges.
        s_plain = sprint(show, MIME"text/plain"(), tn)
        @test occursin("vertices", s_plain)
        @test occursin("edge", s_plain)
        s_default = sprint(show, tn)
        @test occursin("vertices", s_default)

        # `setindex!` for edges is unimplemented.
        e = first(edges(tn))
        @test_throws ErrorException tn[e] = randn(2, 2)
        @test_throws ErrorException tn[src(e) => dst(e)] = randn(2, 2)
    end

    # ---------------------------------------------------------------------------
    # `linkaxes` / `linknames` on a TensorNetwork
    # ---------------------------------------------------------------------------
    @testset "linkaxes / linknames" begin
        g = named_grid((3,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        e = first(edges(tn))
        p = src(e) => dst(e)

        li = linkinds(tn, e)
        la_e = ITensorNetworksNext.linkaxes(tn, e)
        la_p = ITensorNetworksNext.linkaxes(tn, p)
        @test la_e == la_p
        @test length(la_e) == length(li)

        ln_e = ITensorNetworksNext.linknames(tn, e)
        ln_p = ITensorNetworksNext.linknames(tn, p)
        @test ln_e == ln_p
        @test length(ln_e) == length(li)
    end

    # ---------------------------------------------------------------------------
    # expression-shape predicates
    # ---------------------------------------------------------------------------
    @testset "is_setindex!_expr / is_assignment_expr / is_getindex_expr" begin
        @test ITensorNetworksNext.is_setindex!_expr(:(a[1] = 2))
        @test !ITensorNetworksNext.is_setindex!_expr(:(a[1]))
        @test !ITensorNetworksNext.is_setindex!_expr(:(a + b))
        @test !ITensorNetworksNext.is_setindex!_expr(42)

        @test ITensorNetworksNext.is_assignment_expr(:(x = 1))
        @test !ITensorNetworksNext.is_assignment_expr(:(x + 1))
        @test !ITensorNetworksNext.is_assignment_expr(42)

        @test ITensorNetworksNext.is_getindex_expr(:(a[1]))
        @test !ITensorNetworksNext.is_getindex_expr(:(a + 1))
        @test !ITensorNetworksNext.is_getindex_expr(42)
    end

    # ---------------------------------------------------------------------------
    # `add_missing_edges!`: no-op on a well-formed network.
    # ---------------------------------------------------------------------------
    @testset "add_missing_edges!" begin
        g = named_grid((2, 2))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        es_before = collect(edges(tn))
        ITensorNetworksNext.add_missing_edges!(tn)
        @test issetequal(edges(tn), es_before)

        v = first(vertices(tn))
        ITensorNetworksNext.add_missing_edges!(tn, v)
        @test issetequal(edges(tn), es_before)
    end

    # ---------------------------------------------------------------------------
    # `TensorNetwork` constructor / copy / convert variants and `rem_edge!`
    # ---------------------------------------------------------------------------
    @testset "TensorNetwork copy / convert / rem_edge!" begin
        g = named_grid((3,))
        s = Dict(v => Index(2) for v in vertices(g))
        tn = TensorNetwork(g) do v
            return randn(s[v])
        end

        # `TensorNetwork(tensors)` infers the graph from shared indices.
        link = Index(2)
        A = randn(s[(1,)], link)
        B = randn(s[(2,)], link)
        tensors = Dictionary([(1,), (2,)], [A, B])
        tn_inferred = TensorNetwork(tensors)
        @test tn_inferred isa TensorNetwork
        @test issetequal(vertices(tn_inferred), [(1,), (2,)])
        @test ne(tn_inferred) == 1

        # `copy` produces an independent TensorNetwork.
        tn2 = copy(tn)
        @test tn2 isa TensorNetwork
        @test issetequal(vertices(tn2), vertices(tn))
        @test issetequal(edges(tn2), edges(tn))
        @test vertex_data(tn2) !== vertex_data(tn)

        # `TensorNetwork(tn)` and `TensorNetwork{V}(tn)` (same V) call `copy`.
        tn3 = TensorNetwork(tn)
        @test tn3 isa TensorNetwork
        @test issetequal(vertices(tn3), vertices(tn))

        V = GraphsExtensions.vertextype(tn)
        tn4 = TensorNetwork{V}(tn)
        @test tn4 isa TensorNetwork
        @test issetequal(vertices(tn4), vertices(tn))

        # `TensorNetwork{V}(tn)` with a different V re-keys vertices.
        tn5 = TensorNetwork{Tuple{Float64}}(tn)
        @test tn5 isa TensorNetwork
        @test all(v -> v isa Tuple{Float64}, vertices(tn5))

        # `rem_edge!` returns false for an absent edge.
        bad_edge = (1,) => (3,)
        @test !Graphs.rem_edge!(tn, bad_edge)

        # `rem_edge!` on an edge with shared inds throws.
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn_link = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end
        e = first(edges(tn_link))
        @test_throws ArgumentError Graphs.rem_edge!(tn_link, e)
    end

    # ---------------------------------------------------------------------------
    # `induced_subgraph_from_vertices` for TensorNetwork
    # ---------------------------------------------------------------------------
    @testset "TensorNetwork induced_subgraph_from_vertices" begin
        g = named_grid((3,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        sub_vs = [(1,), (2,)]
        subtn, _ = NamedGraphs.induced_subgraph_from_vertices(tn, sub_vs)
        @test subtn isa TensorNetwork
        @test issetequal(vertices(subtn), sub_vs)
    end

    # ---------------------------------------------------------------------------
    # `BeliefPropagationCache` constructor variants and message/factor mutators
    # ---------------------------------------------------------------------------
    @testset "BeliefPropagationCache constructors and mutators" begin
        g = named_grid((2, 2))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        # `BeliefPropagationCache(network)` (no callable; cache constructed).
        bpc1 = BeliefPropagationCache(tn)
        @test bpc1 isa BeliefPropagationCache
        @test length(factors(bpc1)) == nv(tn)

        # `BeliefPropagationCache(callable, network)`
        bpc2 = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end
        @test length(messages(bpc2)) == 2 * length(edges(g))

        # `copy` is independent of the source.
        bpc_copy = copy(bpc2)
        @test bpc_copy isa BeliefPropagationCache
        @test length(messages(bpc_copy)) == length(messages(bpc2))

        # `setmessage!` and `setfactor!` write through the cache.
        e = first(edges(bpc2))
        new_msg = ones(Tuple(linkinds(tn, e))) .* 2.0
        ITensorNetworksNext.setmessage!(bpc2, e, new_msg)
        @test message(bpc2, e) == new_msg

        v = first(vertices(bpc2))
        old_factor = factor(bpc2, v)
        new_factor = old_factor .* 2
        ITensorNetworksNext.setfactor!(bpc2, v, new_factor)
        @test factor(bpc2, v) == new_factor

        # `setmessages!` accepts a mapping and updates entries.
        e2 = first(edges(bpc2))
        msg2 = ones(Tuple(linkinds(tn, e2))) .* 3.0
        ITensorNetworksNext.setmessages!(bpc2, Dict(e2 => msg2))
        @test message(bpc2, e2) == msg2

        # `setmessages!(dst, src, edges)` copies messages between caches.
        bpc_dst = BeliefPropagationCache(tn) do edge
            return zeros(Tuple(linkinds(tn, edge)))
        end
        e3 = first(edges(bpc2))
        ITensorNetworksNext.setmessages!(bpc_dst, bpc2, [e3])
        @test message(bpc_dst, e3) == message(bpc2, e3)
    end

    # ---------------------------------------------------------------------------
    # AbstractBeliefPropagationCache helpers: vertex/edge/region scalars,
    # incoming_messages, map_messages/map_factors, factor_type, message_type.
    # ---------------------------------------------------------------------------
    @testset "BeliefPropagationCache scalars / mappers" begin
        g = named_grid((2,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        bpc = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end
        bpc = ITensorNetworksNext.beliefpropagation(bpc; maxiter = 1)

        v = first(vertices(bpc))
        e = first(edges(bpc))

        # Vertex/edge/region scalars.
        vs = ITensorNetworksNext.vertex_scalar(bpc, v)
        es = ITensorNetworksNext.edge_scalar(bpc, e)
        @test vs isa Number
        @test es isa Number

        rs = ITensorNetworksNext.region_scalar(bpc, [v, e])
        @test rs ≈ vs * es

        # `incoming_messages` excludes specified edges.
        in_msgs = ITensorNetworksNext.incoming_messages(bpc, v)
        in_msgs_filtered = ITensorNetworksNext.incoming_messages(
            bpc, v; ignore_edges = [reverse(e)]
        )
        @test length(in_msgs_filtered) <= length(in_msgs)

        # `factor_type` / `message_type` resolve to concrete types.
        @test factor_type(bpc) isa Type
        @test message_type(bpc) isa Type

        # `map_messages` and `map_factors` produce independent caches.
        bpc_doubled = ITensorNetworksNext.map_messages(m -> 2 .* m, bpc)
        @test message(bpc_doubled, e) ≈ 2 .* message(bpc, e)

        bpc_scaled = ITensorNetworksNext.map_factors(f -> f .* 2, bpc)
        for vv in vertices(bpc_scaled)
            @test factor(bpc_scaled, vv) ≈ factor(bpc, vv) .* 2
        end

        # `adapt_factors` and `adapt_messages` should at least be callable.
        @test ITensorNetworksNext.adapt_factors(identity, bpc) isa BeliefPropagationCache
        @test ITensorNetworksNext.adapt_messages(identity, bpc) isa BeliefPropagationCache
    end

    # ---------------------------------------------------------------------------
    # `logscalar` branches: complex-promotion path and zero denominator.
    # ---------------------------------------------------------------------------
    @testset "logscalar special branches" begin
        g = named_grid((2,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end
        bpc = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end
        bpc = ITensorNetworksNext.beliefpropagation(bpc; maxiter = 1)

        # Negate one factor so the numerator product becomes negative,
        # forcing a complex promotion in `logscalar`.
        v = first(vertices(bpc))
        ITensorNetworksNext.setfactor!(bpc, v, -1 .* factor(bpc, v))
        @test ITensorNetworksNext.logscalar(bpc) isa Number

        # Zero out a message so a denominator term becomes zero -> -Inf.
        bpc_zero = BeliefPropagationCache(tn) do edge
            return zeros(Tuple(linkinds(tn, edge)))
        end
        @test ITensorNetworksNext.logscalar(bpc_zero) == -Inf
    end

    # ---------------------------------------------------------------------------
    # `induced_subgraph_bpcache` / induced_subgraph_from_vertices on a BPCache.
    # ---------------------------------------------------------------------------
    @testset "BeliefPropagationCache induced_subgraph" begin
        g = named_grid((3,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end
        bpc = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end

        sub_vs = [(1,), (2,)]
        subbpc = subgraph(bpc, sub_vs)
        @test subbpc isa BeliefPropagationCache
        @test issetequal(vertices(subbpc), sub_vs)
        @test has_edge(subbpc, (1,) => (2,))
    end

    # ---------------------------------------------------------------------------
    # `forest_cover_edge_sequence` returns a sequence covering a tree.
    # ---------------------------------------------------------------------------
    @testset "forest_cover_edge_sequence" begin
        g = named_comb_tree((3, 2))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end
        bpc = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end

        seq = ITensorNetworksNext.forest_cover_edge_sequence(bpc)
        @test eltype(seq) <: AbstractEdge
        @test !isempty(seq)
    end

    # ---------------------------------------------------------------------------
    # Belief propagation: `select_algorithm` errors when `maxiter` is required.
    # ---------------------------------------------------------------------------
    @testset "beliefpropagation select_algorithm error" begin
        # 2x2 grid: not a tree, so `maxiter` cannot be defaulted.
        g = named_grid((2, 2))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end
        bpc = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end
        @test_throws ArgumentError ITensorNetworksNext.select_algorithm(
            ITensorNetworksNext.beliefpropagation, bpc; maxiter = nothing
        )
    end

    # ---------------------------------------------------------------------------
    # `iterate_diff` and `SimpleMessageUpdate.getproperty(:kwargs)` path.
    # ---------------------------------------------------------------------------
    @testset "iterate_diff and SimpleMessageUpdate kwargs" begin
        g = named_grid((2,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        bpc1 = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end
        bpc2 = BeliefPropagationCache(tn) do edge
            return ones(Tuple(linkinds(tn, edge)))
        end

        # Identical caches: diff should be ~0.
        @test ITensorNetworksNext.iterate_diff(bpc1, bpc2) ≈ 0 atol = 1.0e-10

        # `SimpleMessageUpdate.getproperty(:kwargs)` returns the NamedTuple.
        edge = first(edges(bpc1))
        upd = ITensorNetworksNext.SimpleMessageUpdate(edge; normalize = false)
        @test upd.kwargs isa NamedTuple
        # Forwarded properties still work (`getfield(:kwargs)` then property).
        @test upd.normalize == false
    end

    # ---------------------------------------------------------------------------
    # `contract_network`: unknown algorithm error and `left_associative` order.
    # ---------------------------------------------------------------------------
    @testset "contract_network error / left_associative" begin
        g = named_grid((2,))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        @test_throws ArgumentError contract_network(tn; alg = Algorithm"unknown_alg"())

        # `contraction_order` for `left_associative` algorithm.
        order = ITensorNetworksNext.contraction_order(tn; alg = Algorithm"left_associative"())
        @test order isa LazyNamedDimsArray
    end

    # ---------------------------------------------------------------------------
    # `dmrg`: thin wrappers and unimplemented `EigsolveRegion` step.
    # ---------------------------------------------------------------------------
    @testset "dmrg wrappers" begin
        operator = "operator"
        init = "init"
        nsweeps = 2
        regions = ["region1"]
        algorithm = ITensorNetworksNext.select_algorithm(
            dmrg, operator, init; nsweeps, regions, maxdim = 10
        )

        # `dmrg(operator, algorithm, state)` errors deep in `EigsolveRegion`'s solve!.
        @test_throws Exception dmrg(operator, algorithm, init)

        # `dmrg(operator, state; ...)` builds the algorithm internally; same expected error.
        @test_throws Exception dmrg(operator, init; nsweeps, regions, maxdim = 10)

        # The `EigsolveRegion`-specific `solve!` errors directly.
        region = EigsolveRegion("region"; maxdim = 10)
        problem = ITensorNetworksNext.EigenProblem(operator)
        state = AI.initialize_state(problem, region; iterate = init)
        @test_throws ErrorException AI.solve!(problem, region, state)
    end

    # ---------------------------------------------------------------------------
    # `LazyNamedDimsArrays`: error paths in lazy interface.
    # ---------------------------------------------------------------------------
    @testset "LazyNamedDimsArrays interface error paths" begin
        a = nameddims(randn(2, 2), (:i, :j))
        la = lazy(a)

        # `getindex_lazy` errors on expressions, but works on a leaf.
        @test la[1, 1] == a[1, 1]
        expr = la * lazy(nameddims(randn(2, 2), (:j, :k)))
        @test_throws ErrorException expr[1, 1]

        # `denamed` works on a leaf, errors on non-leaf.
        @test denamed(la) == denamed(a)
        @test_throws ErrorException LazyNamedDimsArrays.denamed_lazy(expr)

        # `dimnames` and `inds` on a `Mul`.
        @test issetequal(dimnames(expr), [:i, :k])
        @test length(inds(expr)) == 2

        # Equality and hash branches.
        la2 = lazy(a)
        @test la == la2
        @test isequal(la, la2)
        @test !(la == expr)             # leaf vs expression
        @test !isequal(la, expr)        # leaf vs expression
        @test hash(la) == hash(la2)

        # `mul_lazy(a)` on a leaf wraps it in a `Mul`.
        wrapped = *(la)
        @test ismul(wrapped)
        @test arguments(wrapped) == [la]

        # `mul_lazy(a)` on a Mul returns it unchanged.
        @test *(expr) == expr

        # `mul_lazy(a, b; flatten=true)` flattens the arguments.
        expr3 = lazy(nameddims(randn(2, 2), (:k, :l)))
        flat = LazyNamedDimsArrays.mul_lazy(expr, expr3; flatten = true)
        @test ismul(flat)
        @test length(arguments(flat)) == 3

        # Number * Number short-circuit.
        @test LazyNamedDimsArrays.mul_lazy(2, 3) == 6

        # Unsupported ops error.
        @test_throws ErrorException la + la2
        @test_throws ErrorException la - la2
        @test_throws ErrorException -la
        @test_throws ErrorException la / 2
        @test_throws ErrorException 2 * la
        @test_throws ErrorException la * 2

        # `maketerm` for non-`*` head errors.
        @test_throws ErrorException LazyNamedDimsArrays.maketerm_lazy(
            LazyNamedDimsArray, +, [la, la2], nothing
        )

        # `parenttype` resolution.
        @test parenttype(LazyNamedDimsArray) === AbstractNamedDimsArray
        @test parenttype(LazyNamedDimsArray{Float64}) === AbstractNamedDimsArray{Float64}
        @test parenttype(typeof(la)) === typeof(a)
    end

    # ---------------------------------------------------------------------------
    # Lazy broadcasting (linear ops only; arbitrary ops error).
    # ---------------------------------------------------------------------------
    @testset "Lazy broadcasting" begin
        a = nameddims(randn(2, 2), (:i, :j))
        la, la2 = lazy(a), lazy(a)
        style = LazyNamedDimsArrays.LazyNamedDimsArrayStyle()

        # Broadcasted linear ops route through `+, -, *, /, unary -`,
        # all of which themselves error in the lazy framework.
        @test_throws ErrorException Base.Broadcast.broadcasted(style, +, la, la2)
        @test_throws ErrorException Base.Broadcast.broadcasted(style, -, la, la2)
        @test_throws ErrorException Base.Broadcast.broadcasted(style, *, 2.0, la)
        @test_throws ErrorException Base.Broadcast.broadcasted(style, *, la, 2.0)
        @test Base.Broadcast.broadcasted(style, *, 2.0, 3.0) == 6.0
        @test_throws ErrorException Base.Broadcast.broadcasted(style, /, la, 2.0)
        @test_throws ErrorException Base.Broadcast.broadcasted(style, -, la)

        # Arbitrary functions error explicitly.
        @test_throws ErrorException Base.Broadcast.broadcasted(style, sin, la)
    end

    # ---------------------------------------------------------------------------
    # `SymbolicArray`: getindex/setindex! errors, permutedims, show, printnode.
    # ---------------------------------------------------------------------------
    @testset "SymbolicArray operations" begin
        sa = SymbolicArray(:x, (Base.OneTo(2), Base.OneTo(3)))
        @test size(sa) == (2, 3)

        # Indexing errors.
        @test_throws ErrorException sa[1, 1]
        @test_throws ErrorException (sa[1, 1] = 0)

        # `permutedims`.
        pa = permutedims(sa, (2, 1))
        @test size(pa) == (3, 2)

        # `show` writes the symbolic name.
        s_plain = sprint(show, MIME"text/plain"(), sa)
        @test occursin("x", s_plain)
        s_default = sprint(show, sa)
        @test occursin("SymbolicArray", s_default)

        # `printnode` writes the symbolic name.
        s_node = sprint(AbstractTrees.printnode, sa)
        @test occursin("x", s_node)
    end

    # ---------------------------------------------------------------------------
    # `SymbolicNamedDimsArray`: equality and printnode with non-zero ndims.
    # ---------------------------------------------------------------------------
    @testset "SymbolicNamedDimsArray equality / printnode" begin
        i, j = namedoneto.(2, (:i, :j))
        sa = symnameddims(:a, (i, j))
        sa2 = symnameddims(:a, (i, j))
        sa_perm = symnameddims(:a, (j, i))
        sa_other = symnameddims(:b, (i, j))

        # Equality: same name + same dimnames (any order) -> equal.
        @test unwrap(sa) == unwrap(sa2)
        @test unwrap(sa) == unwrap(sa_perm)
        @test unwrap(sa) != unwrap(sa_other)

        # `printnode` on a non-scalar prints both name and dims.
        s_node = sprint(AbstractTrees.printnode, unwrap(sa))
        @test occursin("a", s_node)
        @test occursin("[", s_node)
    end

    # ---------------------------------------------------------------------------
    # `evaluation_time_complexity` / `flatten_expression` / `optimize_evaluation_order`
    # ---------------------------------------------------------------------------
    @testset "LazyNamedDimsArrays evaluation_order" begin
        a = nameddims(randn(3, 3), (:i, :j))
        b = nameddims(randn(3, 3), (:j, :k))
        la, lb = lazy.((a, b))
        expr = la * lb

        # Time complexity for a known mul.
        @test LazyNamedDimsArrays.evaluation_time_complexity(expr) > 0

        # Flatten of a `Mul` of `Mul`s.
        c = nameddims(randn(3, 3), (:k, :i))
        lc = lazy(c)
        nested = (la * lb) * lc
        flat = LazyNamedDimsArrays.flatten_expression(nested)
        @test ismul(flat)
        @test length(arguments(flat)) == 3

        # `flatten_expression` is identity on leaves.
        @test LazyNamedDimsArrays.flatten_expression(la) === la

        # `optimize_evaluation_order` on a leaf is identity.
        @test LazyNamedDimsArrays.optimize_evaluation_order(la) === la

        # `optimize_contraction_order` with eager picks an ordering.
        eager = Algorithm"eager"()
        flat_expr = LazyNamedDimsArrays.flatten_expression((la * lb) * lc)
        @test LazyNamedDimsArrays.optimize_evaluation_order(eager, flat_expr) isa
            LazyNamedDimsArray

        # Time-complexity for scalar*tensor and tensor*scalar.
        n = nameddims(randn(3, 3), (:i, :j))
        @test LazyNamedDimsArrays.time_complexity(*, 2.0, n) > 0
        @test LazyNamedDimsArrays.time_complexity(*, n, 2.0) > 0

        # Time complexity for elementwise +.
        n2 = nameddims(randn(3, 3), (:i, :j))
        @test LazyNamedDimsArrays.time_complexity(+, n, n2) > 0
    end

    # ---------------------------------------------------------------------------
    # `nameddimsarraysextensions._hash` fallback for non-NamedDimsArray.
    # ---------------------------------------------------------------------------
    @testset "_hash fallback" begin
        @test LazyNamedDimsArrays._hash(42, UInt64(0)) == hash(42, UInt64(0))
        @test LazyNamedDimsArrays._hash("x", UInt64(0)) == hash("x", UInt64(0))
    end

    # ---------------------------------------------------------------------------
    # `generic_map` for arrays / dicts / sets.
    # ---------------------------------------------------------------------------
    @testset "generic_map" begin
        @test LazyNamedDimsArrays.generic_map(x -> x + 1, [1, 2, 3]) == [2, 3, 4]

        d = Dict(:a => 1, :b => 2)
        md = LazyNamedDimsArrays.generic_map(x -> x * 10, d)
        @test md isa Dict
        @test md[:a] == 10
        @test md[:b] == 20

        ms = LazyNamedDimsArrays.generic_map(x -> x * 2, Set([1, 2, 3]))
        @test ms == Set([2, 4, 6])
    end

    # ---------------------------------------------------------------------------
    # `Mul` core hooks.
    # ---------------------------------------------------------------------------
    @testset "Mul / Applied basics" begin
        a = lazy(nameddims(randn(2, 2), (:i, :j)))
        b = lazy(nameddims(randn(2, 2), (:j, :i)))
        m = Mul([a, b])

        @test arguments(m) == [a, b]
        @test operation(m) ≡ *
        @test iscall(m)
        @test isexpr(m)
        @test head(m) ≡ *

        # `show` for an `Applied` writes parens-joined arguments.
        @test occursin("*", sprint(show, m))

        # Hashing of equal `Mul`s.
        m2 = Mul([a, b])
        @test hash(m) == hash(m2)
    end

    # ---------------------------------------------------------------------------
    # AlgorithmsInterfaceExtensions: `NonIterativeAlgorithm` fallback `solve!`.
    # ---------------------------------------------------------------------------
    @testset "NonIterativeAlgorithm fallback solve!" begin
        problem = _DummyProblem()
        algorithm = _DummyNonIter()
        state = AI.initialize_state(problem, algorithm; iterate = [0.0])
        @test_throws Exception AI.solve!(problem, algorithm, state)
    end

    # ---------------------------------------------------------------------------
    # Latent-bug catchers — these tests are currently expected to FAIL.
    # They exercise code paths whose source references variables that aren't
    # defined in the function body. They exist to surface those bugs the next
    # time someone runs the suite, not to lock in current (buggy) behavior.
    # ---------------------------------------------------------------------------
    @testset "siteaxes / sitenames (latent UndefVarError)" begin
        # Build a TN where each tensor has both link indices (one per neighbor)
        # and a "site" index that no neighbor shares.
        g = named_grid((3,))
        site_idx = Dict(v => Index(2) for v in vertices(g))
        link = Dict(e => Index(2) for e in edges(g))
        link = merge(link, Dict(reverse(e) => link[e] for e in edges(g)))
        tn = TensorNetwork(g) do v
            is = (site_idx[v], (link[e] for e in incident_edges(g, v))...)
            return randn(is)
        end

        e = first(edges(tn))

        # Both functions reference `v` inside their `for v′ in neighbors(tn, v)`
        # loop, but `v` is never defined in either body — only `edge` is in
        # scope. Calling them currently throws `UndefVarError(:v)`.
        # The expected (post-fix) behavior is to return a non-empty collection
        # of the site axes / site names at the edge endpoints, so we assert that
        # the call succeeds and returns something sensible.
        sax = ITensorNetworksNext.siteaxes(tn, e)
        @test sax isa AbstractVector || sax isa AbstractSet || sax isa Tuple
        @test !isempty(sax)

        snm = ITensorNetworksNext.sitenames(tn, e)
        @test snm isa AbstractVector || snm isa AbstractSet || snm isa Tuple
        @test !isempty(snm)
    end
end
