using DataGraphs: assigned_edge_data, assigned_vertex_data, underlying_graph, vertex_data
using Graphs: dst, edges, has_edge, ne, nv, src, vertices
using ITensorBase: Index
using ITensorNetworksNext.LazyNamedDimsArrays: LazyNamedDimsArray
using ITensorNetworksNext: TensorNetwork
using NamedGraphs.GraphsExtensions: vertextype
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: AbstractPartitionedGraph, QuotientVertex, departition,
    partitioned_vertices, partitionedgraph, quotient_graph, quotient_graph_type,
    quotientvertices
using NamedGraphs: convert_vertextype, similar_graph
using Test: @test, @test_throws, @testset

@testset "`TensorNetwork`" begin
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
