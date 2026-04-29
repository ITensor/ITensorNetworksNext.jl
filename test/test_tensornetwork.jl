using DataGraphs: assigned_edge_data, assigned_vertex_data, underlying_graph, vertex_data
using Graphs: add_edge!, add_vertex!, dst, edges, edgetype, has_edge, has_vertex,
    is_directed, ne, nv, rem_vertex!, src, vertices
using ITensorBase: Index
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray
using ITensorNetworksNext:
    TensorNetwork, fix_edges!, linkaxes, linkinds, linknames, siteaxes, siteinds, sitenames
using NamedGraphs.GraphsExtensions: incident_edges, subgraph, vertextype
using NamedGraphs.NamedGraphGenerators: named_grid, named_path_graph
using NamedGraphs.PartitionedGraphs: AbstractPartitionedGraph, QuotientVertex, departition,
    partitioned_vertices, partitionedgraph, quotient_graph, quotient_graph_type,
    quotientvertices
using NamedGraphs: convert_vertextype, similar_graph
using Test: @test, @test_throws, @testset

@testset "`TensorNetwork`" begin
    @testset "Basics" begin
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
        @test !is_directed(typeof(tn))

        # `show` MIME and default both succeed and mention vertices/edges.
        s_plain = sprint(show, MIME"text/plain"(), tn)
        @test occursin("vertices", s_plain)
        @test occursin("edge", s_plain)
        s_default = sprint(show, tn)
        @test occursin("vertices", s_default)

        # `setindex!` for edges is intentionally unimplemented.
        e = first(edges(tn))
        @test_throws ErrorException tn[e] = randn(2, 2)
        @test_throws ErrorException tn[src(e) => dst(e)] = randn(2, 2)

        rem_vertex!(tn, (2, 2))
        @test !has_vertex(tn, (2, 2))
        add_vertex!(tn, (2, 2))
        @test has_vertex(tn, (2, 2))
        @test !isassigned(tn, (2, 2))

        # Test `fix_edges!` removes edges where there is no link index
        t = randn(s[(2, 2)])
        tn[(2, 2)] = t
        add_edge!(tn.underlying_graph, (1, 2) => (2, 2))
        fix_edges!(tn, (2, 2))
        @test !has_edge(tn, (1, 2) => (2, 2))
    end

    @testset "link and site functions" begin
        g = named_path_graph(3)
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        s = Dict(v => Index(2) for v in vertices(g))
        tn = TensorNetwork(g) do v
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
        tn = TensorNetwork(g) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        sub_vs = [(1,), (2,)]
        subtn = subgraph(tn, sub_vs)
        @test subtn isa TensorNetwork
        @test issetequal(vertices(subtn), sub_vs)
        @test has_edge(subtn, (1,) => (2,))
    end

    @testset "DataGraphs/NamedGraphs interface" begin
        dims = (3, 3)
        g = named_grid(dims)
        s = Dict(v => Index(2) for v in vertices(g))
        tn = TensorNetwork(g) do v
            return randn(s[v])
        end

        stn = similar_graph(tn)
        @test stn isa TensorNetwork
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
        @test ctn isa TensorNetwork
        @test vertextype(ctn) == Tuple{Float64, Float64}
        @test collect(vertex_data(ctn)) == collect(vertex_data(tn))
    end

    @testset "`PartitionedGraphs`" begin
        dims = (3, 3)
        g = named_grid(dims)
        s = Dict(v => Index(2) for v in vertices(g))
        tn = TensorNetwork(g) do v
            return randn(s[v])
        end

        # Row partition: each partition is one row of the grid.
        row_parts = [[(i, j) for i in 1:dims[1]] for j in 1:dims[2]]

        @testset "default `partitioned_vertices`" begin
            # By default the entire underlying graph is one partition.
            pvs = partitioned_vertices(tn)
            @test length(pvs) == 1
            @test issetequal(only(pvs), vertices(tn))
        end

        @testset "default `quotientvertices`" begin
            qvs = collect(quotientvertices(tn))
            @test length(qvs) == 1
            @test only(qvs) isa QuotientVertex
        end

        @testset "`tn[QuotientVertex(...)]` (default)" begin
            qv = only(collect(quotientvertices(tn)))
            data = tn[qv]
            @test data isa LazyNamedDimsArray
        end

        @testset "`quotient_graph` (default partitioning)" begin
            qtn = quotient_graph(tn)
            @test qtn isa TensorNetwork
            @test nv(qtn) == 1
            @test ne(qtn) == 0
            v = only(collect(vertices(qtn)))
            @test qtn[v] isa LazyNamedDimsArray
        end

        @testset "`quotient_graph_type`" begin
            QT = quotient_graph_type(typeof(tn))
            @test QT <: TensorNetwork
            qtn = quotient_graph(tn)
            @test vertextype(qtn) === vertextype(QT)
        end

        @testset "`partitionedgraph(tn, parts)`" begin
            ptn = partitionedgraph(tn, row_parts)
            @test ptn isa TensorNetwork
            # The set of underlying vertices/edges is preserved.
            @test issetequal(vertices(ptn), vertices(tn))
            @test issetequal(edges(ptn), edges(tn))
            @test nv(ptn) == nv(tn)
            @test ne(ptn) == ne(tn)
            # Vertex data is copied, not aliased.
            @test collect(vertex_data(ptn)) == collect(vertex_data(tn))
            @test vertex_data(ptn) !== vertex_data(tn)
        end

        @testset "`partitioned_vertices` of partitioned tn" begin
            ptn = partitionedgraph(tn, row_parts)
            pvs = partitioned_vertices(ptn)
            @test length(pvs) == dims[2]
            for part in pvs
                @test length(part) == dims[1]
            end
            @test issetequal(reduce(vcat, pvs), vertices(tn))
        end

        @testset "`tn[QuotientVertex(...)]` (partitioned)" begin
            ptn = partitionedgraph(tn, row_parts)
            for qv in quotientvertices(ptn)
                @test ptn[qv] isa LazyNamedDimsArray
            end
        end

        @testset "`quotient_graph` of partitioned tn" begin
            ptn = partitionedgraph(tn, row_parts)
            qtn = quotient_graph(ptn)
            @test qtn isa TensorNetwork
            @test nv(qtn) == dims[2]
            # The row-partitioned grid quotients to a path graph of length `dims[2]`.
            @test ne(qtn) == dims[2] - 1
            for v in vertices(qtn)
                @test qtn[v] isa LazyNamedDimsArray
            end
        end

        @testset "`departition`" begin
            # `departition` on a non-partitioned tn returns itself.
            @test departition(tn) === tn

            # `departition` on a partitioned tn unwraps one layer of partitioning.
            ptn = partitionedgraph(tn, row_parts)
            dtn = departition(ptn)
            @test dtn isa TensorNetwork
            @test issetequal(vertices(dtn), vertices(tn))
            @test issetequal(edges(dtn), edges(tn))
        end
    end
end
