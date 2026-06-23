using DataGraphs:
    DataGraph, assigned_edge_data, assigned_vertex_data, underlying_graph, vertex_data
using Graphs: add_edge!, add_vertex!, dst, edges, edgetype, has_edge, has_vertex,
    is_directed, ne, nv, rem_edge!, rem_vertex!, src, vertices
using ITensorBase: Index, LazyITensor, inds
using ITensorNetworksNext: ITensorNetwork, has_ind, linkaxes, linkinds, linknames, siteaxes,
    siteinds, sitenames, tensornetwork
using NamedGraphs.GraphsExtensions: incident_edges, subgraph, vertextype
using NamedGraphs.NamedGraphGenerators: named_grid, named_path_graph
using NamedGraphs.PartitionedGraphs: AbstractPartitionedGraph, QuotientVertex, departition,
    partitioned_vertices, partitionedgraph, quotient_graph, quotient_graph_type,
    quotientvertices
using NamedGraphs: convert_vertextype, similar_graph
using Test: @test, @test_throws, @testset

@testset "`ITensorNetwork`" begin
    @testset "Basics" begin
        g = named_grid((2, 2))
        tn = tensornetwork(vertices(g)) do _
            return randn(Index(2))
        end

        # `iterate` works (delegates to `vertex_data`).
        @test !isempty(collect(tn))
        # `keys` returns vertices.
        @test issetequal(keys(tn), vertices(tn))
        # `eltype` matches the eltype of the vertex data.
        @test eltype(tn) === eltype(vertex_data(tn))
        # `is_directed` is `false` for AbstractITensorNetwork.
        @test !is_directed(typeof(tn))

        # `show` MIME and default both succeed and mention vertices/edges.
        s_plain = sprint(show, MIME"text/plain"(), tn)
        @test occursin("vertices", s_plain)
        @test occursin("edge", s_plain)
        s_default = sprint(show, tn)
        @test occursin("vertices", s_default)

        # No link indices so should have no edges
        @test ne(tn) == 0

        j = Index(2)
        tn[1, 1] = randn(j)
        tn[2, 1] = randn(j)

        # No link indices so should have no edges
        @test ne(tn) == 1
        @test has_edge(tn, (1, 1) => (2, 1))

        # `setindex!` for edges is intentionally unimplemented.
        e = first(edges(tn))
        @test_throws MethodError tn[e] = randn(2, 2)
        @test_throws MethodError tn[src(e) => dst(e)] = randn(2, 2)

        # `rem_edge!` is intentionally unimplemented.
        @test_throws ErrorException rem_edge!(tn, (1, 1) => (2, 1))

        tn[1, 1] = randn(Index(2))
        tn[2, 1] = randn(Index(2))

        @test !has_edge(tn, (1, 1) => (2, 1))
        @test ne(tn) == 0

        rem_vertex!(tn, (2, 2))
        @test !has_vertex(tn, (2, 2))
        insert!(tn, (2, 2), randn(Index(2)))
        @test has_vertex(tn, (2, 2))
        @test isassigned(tn, (2, 2))

        links = DataGraph(named_path_graph(4))
        i = Index(2)
        j = Index(3)
        k = Index(4)
        links[1 => 2] = i
        links[2 => 3] = j
        links[3 => 4] = k

        tn = tensornetwork(vertices(links)) do v
            indices = map(e -> getindex(links, e), incident_edges(links, v))
            return randn(Tuple(indices))
        end

        @test has_ind(tn, i)
        @test has_ind(tn, j)
        @test has_ind(tn, k)

        ip = Index(2)

        tn[1] = tn[1] * randn((i, ip))
        @test has_ind(tn, ip)
        @test has_ind(tn, j)
        @test has_ind(tn, k)

        @test inds(tn[1]) == (ip,)
        @test inds(tn[2]) == (i, j)
        @test inds(tn[3]) == (j, k)
        @test inds(tn[4]) == (k,)
    end

    @testset "link and site functions" begin
        g = named_path_graph(3)
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        s = Dict(v => Index(2) for v in vertices(g))
        tn = tensornetwork(vertices(g)) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn((s[v], is...))
        end

        E = edgetype(tn)
        @test linkinds(tn, 1 => 2) == [l[E(1 => 2)]]
        @test linkinds(tn, E(1 => 2)) == [l[E(1 => 2)]]

        @test linkaxes(tn, 1 => 2) == [l[E(1 => 2)]]
        @test linkaxes(tn, E(1 => 2)) == [l[E(1 => 2)]]

        @test linknames(tn, 1 => 2) == [l[E(1 => 2)].name]
        @test linknames(tn, E(1 => 2)) == [l[E(1 => 2)].name]

        @test siteinds(tn, 1) == [s[1]]
        @test siteaxes(tn, 2) == [s[2]]
        @test sitenames(tn, 3) == [s[3].name]
    end

    @testset "`subgraph`" begin
        g = named_grid((3,))

        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))

        tn = tensornetwork(vertices(g)) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        sub_vs = [(1,), (2,)]
        subtn = subgraph(tn, sub_vs)
        @test subtn isa ITensorNetwork
        @test issetequal(vertices(subtn), sub_vs)
        @test has_edge(subtn, (1,) => (2,))
    end

    @testset "DataGraphs/NamedGraphs interface" begin
        dims = (3, 3)
        g = named_grid(dims)
        s = Dict(v => Index(2) for v in vertices(g))
        tn = tensornetwork(vertices(g)) do v
            return randn(s[v])
        end

        stn = similar_graph(tn)
        @test stn isa ITensorNetwork
        @test vertices(stn) == vertices(tn)
        @test edges(stn) == edges(tn)
        @test isempty(assigned_vertex_data(stn))
        @test isempty(assigned_edge_data(stn))

        stn = similar_graph(tn, vertices(tn))
        @test vertices(stn) == vertices(tn)
        @test ne(stn) == 0
        @test isempty(assigned_vertex_data(stn))
        @test isempty(assigned_edge_data(stn))

        stn = similar_graph(typeof(tn))
        @test nv(stn) == 0
        @test stn isa typeof(tn)

        stn = similar_graph(typeof(tn), vertices(tn))
        @test nv(stn) == nv(tn)
        @test ne(stn) == 0
        @test stn isa typeof(tn)

        ctn = convert_vertextype(Tuple{Float64, Float64}, tn)
        @test ctn isa ITensorNetwork
        @test vertextype(ctn) == Tuple{Float64, Float64}
        @test collect(vertex_data(ctn)) == collect(vertex_data(tn))
    end
end
